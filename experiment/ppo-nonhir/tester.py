# this file is under mypy's checking
from __future__ import annotations

import heapq
import os

import hydra
import qtz
import torch
import wandb
from ds import *
from model.actor_critic import NonHirActorCritic
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
        rank: int,
    ) -> None:
        self.cost_type = cost_type
        self.ac_net = ac_net
        self.device = device
        self.output_dir = output_dir
        self.rank = rank

    def beam_search(
        self,
        input_graph: quartz.PyGraph,
        topk: int,
        max_eps_len: int,
        name: str,
        budget: int = int(1e6),
    ) -> Tuple[int, float]:

        cur_graph: quartz.PyGraph = input_graph
        cur_cost = get_cost(cur_graph, self.cost_type)
        cur_hash = hash(cur_graph)
        prev_exp = PrevExp(cur_graph, cur_hash, 0, cur_cost, Action(0, 0), 0, 0, 0)

        hash_prevexp: Dict[int, PrevExp] = {cur_hash: prev_exp}
        q: List[PrevExp] = [prev_exp]
        best_graph, best_cost, best_hash, best_time = (
            cur_graph,
            cur_cost,
            cur_hash,
            time.time(),
        )

        with tqdm(
            total=cur_cost,
            desc=f'rank {self.rank} cost reduced',
            bar_format='{desc}: {n}/{total} |{bar}| {elapsed} {postfix}',
        ) as pbar:
            start_time = time.time()
            while len(q) > 0 and budget > 0:
                popped_exp = heapq.heappop(q)
                budget -= 1
                cur_graph = popped_exp.cur_graph

                dgl_graph = cur_graph.to_dgl_graph().to(self.device)
                node_embeds: torch.Tensor = self.ac_net.graph_embedding(dgl_graph)
                node_values: torch.Tensor = self.ac_net.critic(node_embeds).squeeze()
                topk_node_values, topk_nodes = torch.topk(node_values, topk)
                for action_node_value, action_node in zip(topk_node_values, topk_nodes):
                    action_node_embed = node_embeds[action_node]
                    xfer_logits: torch.Tensor = self.ac_net.actor(action_node_embed)
                    av_xfers = cur_graph.available_xfers_parallel(
                        context=qtz.quartz_context,
                        node=cur_graph.get_node_from_id(id=action_node),
                    )
                    av_mask = torch.BoolTensor([0] * qtz.quartz_context.num_xfers).to(
                        self.device
                    )
                    av_mask[av_xfers] = True
                    xfer_logits[~av_mask] -= 1e10
                    topk_xfer_logits, topk_xfers = torch.topk(xfer_logits, k=topk)
                    for action_xfer_logit, action_xfer in zip(
                        topk_xfer_logits, topk_xfers
                    ):
                        action = Action(int(action_node), int(action_xfer))
                        (
                            next_graph,
                            next_nodes,
                        ) = cur_graph.apply_xfer_with_local_state_tracking(
                            xfer=qtz.quartz_context.get_xfer_from_id(id=action.xfer),
                            node=cur_graph.get_node_from_id(id=action.node),
                            eliminate_rotation=qtz.has_parameterized_gate,
                        )
                        if next_graph is not None:
                            next_hash = hash(next_graph)
                            if (
                                next_hash not in hash_prevexp
                                and popped_exp.depth + 1 < max_eps_len
                            ):
                                next_cost = get_cost(next_graph, self.cost_type)
                                action_node_value = float(action_node_value)
                                action_xfer_logit = float(action_xfer_logit)
                                prev_exp = PrevExp(
                                    next_graph,
                                    next_hash,
                                    popped_exp.cur_hash,
                                    next_cost,
                                    action,
                                    action_node_value,
                                    action_xfer_logit,
                                    popped_exp.depth + 1,
                                )
                                hash_prevexp[next_hash] = prev_exp
                                heapq.heappush(q, prev_exp)

                                if next_cost < best_cost:
                                    pbar.update(best_cost - next_cost)
                                    best_graph, best_cost, best_hash, best_time = (
                                        next_graph,
                                        next_cost,
                                        next_hash,
                                        time.time(),
                                    )
                                    time_delta_sec = best_time - start_time
                                    printfl(
                                        f'rank {self.rank} Better graph with cost {best_cost} is found in {time_delta_sec} s ({sec_to_hms(time_delta_sec)})!'
                                        f' node_value: {action_node_value}'
                                        f' xfer_logits: {action_xfer_logit} ({action_xfer_logit / float(xfer_logits.sum())})'
                                    )
                                # end if
                            # end if
                        # end if
                    # end for xfer
                # end for node
                pbar.set_postfix(
                    {
                        'cur_cost': popped_exp.cur_cost,
                        'best_cost': best_cost,
                        '|q|': len(q),
                        '|hash_prevexp|': len(hash_prevexp),
                        'budget': budget,
                    }
                )
                if budget % 1000 == 0:
                    pbar.refresh()
            # end while
        # end with
        """output seq"""
        out_dir = os.path.join(self.output_dir, 'out_graphs', name)
        os.makedirs(out_dir, exist_ok=True)
        printfl(f'rank {self.rank} saving the path to {out_dir} ...')
        prevexp_list: List[PrevExp] = []

        cur_hash = best_hash
        while cur_hash is not None and cur_hash != 0:
            prev_exp = hash_prevexp[cur_hash]
            prevexp_list.append(prev_exp)
            cur_hash = prev_exp.prev_hash

        for i_step, prev_exp in enumerate(reversed(prevexp_list)):
            file_name = f'{i_step}_{prev_exp.cur_cost}_{prev_exp.action.node}_{prev_exp.action.xfer}.qasm'
            with open(os.path.join(out_dir, file_name), 'w') as f:
                f.write(prev_exp.cur_graph.to_qasm_str())

        return best_cost, best_time - start_time
