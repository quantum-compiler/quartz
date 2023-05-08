# this file is under mypy's checking
from __future__ import annotations

import gc
import heapq
import json
import os
from typing import Dict, List, Optional

import hydra
import qtz
import torch
import wandb
from ds import *
from model.actor_critic import ActorCritic
from natsort import natsorted
from torch.distributions import Categorical
from tqdm import tqdm  # type: ignore
from utils import *

from config.config import *


@dataclass
class PrevExp:
    cur_graph: quartz.PyGraph
    cur_hash: int
    prev_hash: int
    cur_cost: int
    action: Action
    node_value: float
    xfer_value: float
    depth: int

    def __lt__(self, other: PrevExp) -> bool:
        if self.cur_cost < other.cur_cost:
            return True
        elif self.cur_cost > other.cur_cost:
            return False
        else:
            if self.node_value < other.node_value:
                return False
            elif self.node_value > other.node_value:
                return True
            else:
                if self.xfer_value < other.xfer_value:
                    return False
                elif self.xfer_value > other.xfer_value:
                    return True
                else:
                    return True

    def __le__(self, other: PrevExp) -> bool:
        if self.cur_cost <= other.cur_cost:
            return True
        else:
            return False


class Tester:
    def __init__(
        self,
        cost_type: CostType,
        ac_net: ActorCritic,
        device: torch.device,
        output_dir: str,
        sync_tuning_dir: bool,
        # rank: int,
        hit_rate: float,
        batch_size: int,
        max_loss_tolerance: float,
        max_search_sec: float,
        vmem_perct_limit: float,
    ) -> None:
        self.cost_type = cost_type
        self.ac_net = ac_net
        self.device = device
        self.output_dir = output_dir
        self.sync_tuning_dir = sync_tuning_dir
        # self.rank = rank
        self.hit_rate = hit_rate
        self.batch_size = batch_size
        self.max_loss_tolerance = max_loss_tolerance
        self.max_search_sec = max_search_sec
        self.vmem_perct_limit = vmem_perct_limit

    def search(
        self,
        input_graphs: Dict[str, quartz.PyGraph],
    ) -> None:
        # from pympler import muppy
        # from pympler import summary
        # sum0 = summary.summarize(muppy.get_objects())
        best_graphs = input_graphs
        t_start = time.time()
        while True:
            # gc.collect()
            # summary.print_(summary.get_diff(sum0, summary.summarize(muppy.get_objects())))
            printfl(
                f"==== Tester: mem usage: {cur_proc_vmem_perct()}, duration: {time.time() - t_start:0f} sec"
            )
            """update model from ckpt"""
            latest_info_path = os.path.join(self.output_dir, "ckpts", "latest.json")
            if self.sync_tuning_dir and os.path.exists(latest_info_path):
                while True:
                    try:
                        with open(latest_info_path, "r") as f:
                            latest_info = json.load(f)
                        break
                    except json.decoder.JSONDecodeError as e:
                        time.sleep(1)
                        continue
                loss: float = latest_info["loss"]
                ckpt_path: str = latest_info["path"]
                if loss < self.max_loss_tolerance:
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    model_state_dict = ckpt["model_state_dict"]
                    self.ac_net.load_state_dict(model_state_dict)
                    printfl(f"Test: use {loss = } , {ckpt_path = }")
                else:
                    printfl(f"Test: ignore {loss = } , {ckpt_path = }")
            """search better circs"""
            for circ_name, circ in best_graphs.items():
                ret_circ = self.random_search(circ_name, circ)
                if ret_circ:  # better
                    best_graphs[circ_name] = ret_circ
                    printfl(
                        f"Test: better circ: cost = {get_cost(ret_circ, self.cost_type)}"
                    )
            """dump best circs"""
            if self.sync_tuning_dir:
                sync_dir = os.path.join(self.output_dir, "sync_dir")
                os.makedirs(sync_dir, exist_ok=True)
                best_info = [
                    {
                        "name": name,
                        "best_cost": get_cost(circ, self.cost_type),
                        "qasm": circ.to_qasm_str(),
                    }
                    for name, circ in best_graphs.items()
                ]
                with open(os.path.join(sync_dir, f"best_info_search.json"), "w") as f:
                    json.dump(best_info, fp=f, indent=2)
        # end while

    @torch.no_grad()
    def random_search(
        self, circ_name: str, circ: quartz.PyGraph
    ) -> Optional[quartz.PyGraph]:
        start_circ_to_vis: Dict[quartz.PyGraph, int] = {circ: 1}
        circ_to_softmask: Dict[quartz.Graph, torch.Tensor] = {}
        circ_to_hardmask: Dict[quartz.Graph, torch.Tensor] = {}

        def sample_start_circs(k: int) -> List[quartz.PyGraph]:
            circs, weights = [], []
            for circ, vis in start_circ_to_vis.items():
                circs.append(circ)
                weights.append(1 / vis)
            sampled_circs = random.choices(circs, weights=weights, k=k)
            return sampled_circs

        best_circ = circ
        cur_circs: List[quartz.PyGraph] = [best_circ] * self.batch_size
        traj_lens: List[int] = [0] * self.batch_size
        expand_times: int = 0
        start_time = time.time()
        while time.time() - start_time < self.max_search_sec:
            """fetch best circ from tuning"""
            if False:
                best_info_dir = os.path.join(self.output_dir, "sync_dir")
                best_info_path = os.path.join(best_info_dir, "best_info_0.json")
                if self.sync_tuning_dir and os.path.exists(best_info_path):
                    while True:
                        try:
                            with open(best_info_path, "r") as f:
                                best_info: list = json.load(f)
                            break
                        except json.decoder.JSONDecodeError as e:
                            time.sleep(1)
                            continue
                    for circ_best_info in best_info:
                        if circ_best_info["name"] == circ_name and circ_best_info[
                            "best_cost"
                        ] < get_cost(best_circ, self.cost_type):
                            better_circ = qtz.qasm_to_graph(circ_best_info["qasm"])
                            printfl(
                                f'Test: get better circ from tuning, cost = {circ_best_info["best_cost"]}'
                            )
                            return better_circ
            cir_dir = os.path.join(self.output_dir, circ_name)
            if self.sync_tuning_dir and os.path.exists(cir_dir):
                cir_seq_dirs = natsorted(os.listdir(cir_dir))
                if len(cir_seq_dirs):
                    better_seq = cir_seq_dirs[0]
                    _better_cost = int(better_seq.split("_")[0])
                    better_seq_dir = os.path.join(
                        cir_dir, better_seq
                    )  # outputs/2023-02-13/11-55-57/gf2^8_mult/859_0
                    better_cost = 0
                    while better_cost != _better_cost:
                        circ_names = natsorted(os.listdir(better_seq_dir), reverse=True)
                        if circ_names:
                            better_circ_name = circ_names[0]  # 74_859_0_0_0.qasm
                            better_cost = int(better_circ_name.split("_")[1])
                    better_circ_path = os.path.join(better_seq_dir, better_circ_name)
                    while better_cost < get_cost(best_circ, self.cost_type):
                        time.sleep(2)
                        try:
                            with open(better_circ_path) as f:
                                better_qasm = f.read()
                            better_circ = qtz.qasm_to_graph(better_qasm)
                            assert (
                                get_cost(better_circ, self.cost_type) == better_cost
                            ), f"{better_cost = }, {get_cost(better_circ, self.cost_type) = }"
                            printfl(
                                f"Test: get better circ from tuning, cost = {better_cost}"
                            )
                            # del start_circ_to_vis, circ_to_softmask, circ_to_hardmask
                            return better_circ
                        except Exception as e:
                            printfl(f"Error when reading better circ from tuning: {e}")
                            continue
            """step forward"""
            cur_circs = self.random_expand(
                cur_circs,
                circ_to_softmask,
                circ_to_hardmask,
                get_cost(best_circ, self.cost_type) + 6,
            )
            expand_times += 1
            """prepare for next step"""
            indices_to_sample: List[int] = []
            tmp_best_circ = best_circ
            for i, cur_circ in enumerate(cur_circs):
                if (
                    cur_circ is None
                ):  # or get_cost(cur_circ, self.cost_type) > get_cost(best_circ, self.cost_type) + 6:
                    indices_to_sample.append(i)
                    traj_lens[i] = 0
                elif get_cost(cur_circ, self.cost_type) < get_cost(
                    tmp_best_circ, self.cost_type
                ):
                    tmp_best_circ = cur_circ
                elif (
                    get_cost(tmp_best_circ, self.cost_type)
                    == get_cost(best_circ, self.cost_type)
                    == get_cost(cur_circ, self.cost_type)
                ):
                    # add circ into starting circ buffer
                    # if vmem_used_perct() > self.vmem_perct_limit and start_circ_to_vis:
                    #     pop_dict_first(start_circ_to_vis)
                    cur_circ_vis = start_circ_to_vis.get(cur_circ, 0)
                    start_circ_to_vis[cur_circ] = cur_circ_vis + 1
                    traj_lens[i] += 1

            if get_cost(tmp_best_circ, self.cost_type) < get_cost(
                best_circ, self.cost_type
            ):
                printfl(
                    f"Test: {expand_times = }, {len(start_circ_to_vis) = }, lasted {time.time() - start_time:.2f} secs"
                )
                printfl(
                    f"Test: get better circ by search, cost = {get_cost(tmp_best_circ, self.cost_type)}"
                )
                # del start_circ_to_vis, circ_to_softmask, circ_to_hardmask
                return tmp_best_circ

            sampled_circs = sample_start_circs(k=len(indices_to_sample))
            for idx, sampled_circs in zip(indices_to_sample, sampled_circs):
                cur_circs[idx] = sampled_circs

            if expand_times % 20 == 0:
                printfl(
                    f"Test: best_cost = {get_cost(best_circ, self.cost_type)}, {expand_times = }, {len(start_circ_to_vis) = }, lasted {time.time() - start_time:.2f} secs"
                )
        # end while
        printfl(
            f"Test: did not find better circ in {self.max_search_sec} sec. Try to load new ckpt and restart..."
        )
        del start_circ_to_vis, circ_to_softmask, circ_to_hardmask
        return None

    def random_expand(
        self,
        cur_circs: List[quartz.PyGraph],
        circ_to_soft_valid: Dict[quartz.Graph, torch.Tensor],
        circ_to_hard_valid: Dict[quartz.Graph, torch.Tensor],
        max_cost: int,
    ) -> List[quartz.PyGraph]:
        self.ac_net.eval()
        num_eps = len(cur_circs)
        """compute embeds and use Critic to evaluate each node"""
        b_circs: dgl.DGLGraph = dgl.batch(
            [circuit.to_dgl_graph() for circuit in cur_circs]
        ).to(self.device)
        num_nodes: torch.LongTensor = (
            b_circs.batch_num_nodes()
        )  # (num_graphs, ) assert each elem > 0
        # (batch_num_nodes, embed_dim)
        b_node_embeds: torch.Tensor = self.ac_net.gnn(b_circs)
        # (batch_num_nodes, )
        b_node_values: torch.Tensor = self.ac_net.critic(b_node_embeds).squeeze()
        # list with length num_graphs; each member is a tensor of node values in a graph
        node_values_list: List[torch.Tensor] = torch.split(
            b_node_values, num_nodes.tolist()
        )
        # (num_graphs, max_num_nodes)
        b_node_values_pad = nn.utils.rnn.pad_sequence(
            node_values_list,
            batch_first=True,
            padding_value=-math.inf,
        )
        """fetch node masks"""
        nodes_valid_list: List[torch.Tensor] = []
        for i_circ, circ in enumerate(cur_circs):
            # if vmem_used_perct() > self.vmem_perct_limit:
            #     if circ_to_soft_valid:
            #         pop_dict_first(circ_to_soft_valid)
            #     if circ_to_hard_valid:
            #         pop_dict_first(circ_to_hard_valid)

            soft_valid_pad = torch.zeros(
                (b_node_values_pad.shape[-1],), dtype=torch.bool
            ).to(self.device)
            soft_valid = circ_to_soft_valid.get(
                circ, torch.ones((circ.gate_count,), dtype=torch.bool).to(self.device)
            )
            circ_to_soft_valid[circ] = soft_valid
            soft_valid_pad[: circ.gate_count] = soft_valid

            hard_valid_pad = torch.zeros(
                (b_node_values_pad.shape[-1],), dtype=torch.bool
            ).to(self.device)
            hard_valid = circ_to_hard_valid.get(
                circ, torch.ones((circ.gate_count,), dtype=torch.bool).to(self.device)
            )
            circ_to_hard_valid[circ] = hard_valid
            hard_valid_pad[: circ.gate_count] = hard_valid

            valid = soft_valid_pad & hard_valid_pad
            if torch.any(valid):
                nodes_valid_list.append(valid)
            else:
                nodes_valid_list.append(hard_valid_pad)
        # end for
        b_nodes_valid = torch.stack(nodes_valid_list)  # (num_graphs, max_num_nodes)
        """sample node by softmax with temperature for each graph"""
        temperatures = 1 / (
            torch.log(self.hit_rate * (num_nodes - 1) / (1 - self.hit_rate))
        )
        node_logits = masked_softmax(
            b_node_values_pad / temperatures.unsqueeze(1), b_nodes_valid
        )
        b_sampled_nodes = torch.multinomial(node_logits, 1).flatten()
        action_nodes: List[int] = b_sampled_nodes.tolist()
        node_offsets = torch.zeros(b_sampled_nodes.shape[0], dtype=torch.long).to(
            self.device
        )
        node_offsets[1:] = torch.cumsum(num_nodes, dim=0)[:-1]
        sampled_node_b_ids = b_sampled_nodes + node_offsets
        # (num_graphs, embed_dim)
        sampled_node_embeds = b_node_embeds[sampled_node_b_ids]
        """use Actor to evaluate xfers for sampled nodes"""
        # (num_graphs, action_dim)
        xfer_logits: torch.Tensor = self.ac_net.actor(sampled_node_embeds)
        """sample action_xfer with mask"""
        av_xfer_masks = torch.zeros_like(
            xfer_logits, dtype=torch.bool
        )  # device is the same with xfer_logits
        av_xfer_masks = cast(torch.BoolTensor, av_xfer_masks)
        for i_circ, circ in enumerate(cur_circs):
            circ = cur_circs[i_circ]
            av_xfers = circ.available_xfers_parallel(
                context=qtz.quartz_context,
                node=circ.get_node_from_id(id=action_nodes[i_circ]),
            )
            av_xfer_masks[i_circ][av_xfers] = True
            circ_to_soft_valid[circ][action_nodes[i_circ]] = False
            # soft_valid = circ_to_soft_valid.get(
            #     circ, torch.ones((circ.gate_count,), dtype=torch.bool).to(self.device)
            # )
            # soft_valid[action_nodes[i_circ]] = False
            # circ_to_soft_valid[circ] = soft_valid
        # end for
        # (num_graphs, action_dim)
        softmax_xfer_logits = masked_softmax(xfer_logits, av_xfer_masks)
        # NOTE: sample or max ?
        # action_xfers: List[int] = (
        #     torch.multinomial(softmax_xfer_logits, num_samples=1).flatten().tolist()
        # )
        action_xfers: List[int] = torch.argmax(
            softmax_xfer_logits, dim=-1, keepdim=False
        ).tolist()
        next_circs: List[quartz.PyGraph] = []
        for i_circ, circ in enumerate(cur_circs):
            xfer: quartz.PyXfer = qtz.quartz_context.get_xfer_from_id(
                id=action_xfers[i_circ]
            )
            if (
                xfer.is_nop
                or circ.gate_count - xfer.src_gate_count + xfer.dst_gate_count
                > max_cost
            ):
                next_circs.append(None)
                circ_to_hard_valid[circ][action_nodes[i_circ]] = False
                # hard_valid = circ_to_hard_valid.get(
                #     circ, torch.ones((circ.gate_count,), dtype=torch.bool).to(self.device)
                # )
                # hard_valid[action_nodes[i_circ]] = False
                # circ_to_hard_valid[circ] = hard_valid
            else:
                next_circ: quartz.PyGraph = circ.apply_xfer(
                    node=circ.get_node_from_id(id=action_nodes[i_circ]),
                    xfer=xfer,
                    eliminate_rotation=True,
                )
                next_circs.append(next_circ)
        # end for
        return next_circs
