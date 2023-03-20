import math
import os
import random
import time
import warnings
from typing import OrderedDict, cast

import dgl
import hydra
import torch
import torch.nn.functional as F
from model.actor_critic import ActorCritic
from torch.distributions import Categorical
from utils import masked_softmax

import quartz
from config.config import *


class ParallelSearchAgent:
    def __init__(
        self,
        qtz: quartz.QuartzContext,
        circuit: quartz.PyGraph,
        num_workers: int,
        max_cost_ratio: float,
        max_cost_increase: int,
        max_trajectory_len: int,
        actor_critic: ActorCritic,
        hit_rate: float,
        device: torch.device,
    ) -> None:
        self.qtz: quartz.QuartzContext = qtz
        self.input_circuit: quartz.PyGraph = circuit
        self.init_cost: int = circuit.gate_count
        self.min_cost: int = self.init_cost
        self.max_cost_ratio: float = max_cost_ratio
        self.actor_critic: ActorCritic = actor_critic
        self.actor_critic.eval()
        self.num_workers: int = num_workers
        self.hit_rate: float = hit_rate
        self.max_cost_increase: int = max_cost_increase
        self.max_trajectory_len: int = max_trajectory_len
        self.device: torch.device = device

        self.circ_sample_set: set[quartz.PyGraph] = set([circuit])
        self.circ_list: list[quartz.PyGraph] = [circuit]
        self.visit_cnt: dict[quartz.PyGraph, int] = {circuit: 1}
        self.node_soft_masks: dict[quartz.PyGraph, torch.Tensor] = {}
        self.node_hard_masks: dict[quartz.PyGraph, torch.Tensor] = {}
        # self.circ_seen: set[quartz.PyGraph] = set([circuit])

        self.none_count: int = 0

    def expand(self, circuit_list: list[quartz.PyGraph]) -> list[quartz.PyGraph | None]:
        b_circs: dgl.DGLGraph = dgl.batch(
            [circuit.to_dgl_graph() for circuit in circuit_list]
        ).to(self.device)
        num_nodes: torch.Tensor = b_circs.batch_num_nodes()
        b_node_embeds: torch.Tensor = self.actor_critic.gnn(b_circs)
        b_node_values: torch.Tensor = self.actor_critic.critic(b_node_embeds).squeeze()
        node_embeds_list: list[torch.Tensor] = torch.split(
            b_node_embeds, num_nodes.tolist()
        )

        # Sample nodes
        node_values_list: list[torch.Tensor] = torch.split(
            b_node_values, num_nodes.tolist()
        )
        temperatures = 1 / (
            torch.log(self.hit_rate * (num_nodes - 1) / (1 - self.hit_rate))
        )

        # Fetch node masks
        node_masks: list[torch.Tensor] = []
        for circ, node_value in zip(circuit_list, node_values_list):
            if circ not in self.node_soft_masks:
                self.node_soft_masks[circ] = torch.zeros(
                    (circ.gate_count,), dtype=torch.bool
                ).to(self.device)
            if circ not in self.node_hard_masks:
                self.node_hard_masks[circ] = torch.zeros(
                    (circ.gate_count,), dtype=torch.bool
                ).to(self.device)
                # self.node_hard_masks[circ] = node_value < 1e-2
            node_soft_mask = self.node_soft_masks[circ]
            node_hard_mask = self.node_hard_masks[circ]
            if (node_hard_mask | node_soft_mask).all():
                node_masks.append(node_hard_mask)
            else:
                node_masks.append(node_hard_mask | node_soft_mask)

        nodes: list[int] = [
            torch.multinomial(
                F.softmax(node_values / temperature - node_mask * 1e10, dim=0), 1
            )[0].item()
            for node_values, temperature, node_mask in zip(
                node_values_list, temperatures, node_masks
            )
        ]

        # Maintain node masks
        for node, circ in zip(nodes, circuit_list):
            self.node_soft_masks[circ][node] = True

        # Construct masks
        mask: torch.Tensor = torch.zeros(
            (self.num_workers, self.qtz.num_xfers), dtype=torch.bool
        )
        for i, (n, circ) in enumerate(zip(nodes, circuit_list)):
            appliable_xfers: list[int] = circ.available_xfers_parallel(
                context=self.qtz, node=circ.get_node_from_id(id=n)
            )
            mask[i][appliable_xfers] = True

        # Select xfers
        selected_node_embeds: list[torch.Tensor] = [
            node_embeds[node] for node_embeds, node in zip(node_embeds_list, nodes)
        ]
        b_selected_node_embeds: torch.Tensor = torch.stack(selected_node_embeds, dim=0)
        b_xfer_logits: torch.Tensor = self.actor_critic.actor(b_selected_node_embeds)
        b_xfer_probs: torch.Tensor = masked_softmax(b_xfer_logits, mask)
        xfers: list[int] = torch.max(b_xfer_probs, dim=1)[1].tolist()

        # Construct new circuits
        new_circ_list: list[quartz.PyGraph] = []
        for i, (n, x, circ) in enumerate(zip(nodes, xfers, circuit_list)):
            xfer: quartz.PyXfer = self.qtz.get_xfer_from_id(id=x)
            if xfer.is_nop:
                new_circ_list.append(None)
                self.node_hard_masks[circ][n] = True
            elif (
                circ.gate_count - xfer.src_gate_count + xfer.dst_gate_count
                > self.min_cost + self.max_cost_increase
            ):
                new_circ_list.append(None)
                self.node_hard_masks[circ][n] = True
            else:
                new_circ: quartz.PyGraph = circ.apply_xfer(
                    xfer=xfer, node=circ.get_node_from_id(id=n), eliminate_rotation=True
                )
                new_circ_list.append(new_circ)

                # If the circuit is large, it is not added to data structures
                if new_circ.gate_count > self.min_cost:
                    continue

                if new_circ.gate_count < self.min_cost:
                    # New min is found, update data structures
                    self.min_cost = new_circ.gate_count
                    print(f'New min gate count: {self.min_cost}')
                    self.circ_sample_set = set([new_circ])
                    self.circ_list = [new_circ]
                    self.visit_cnt = {new_circ: 1}
                    return [None] * self.num_workers
                elif new_circ not in self.circ_sample_set:
                    self.circ_sample_set.add(new_circ)
                    self.circ_list.append(new_circ)
                    self.visit_cnt[new_circ] = 1
                else:
                    self.visit_cnt[new_circ] += 1

        return new_circ_list

    def run(self):
        start = time.time()
        expansion_cnt: int = 0
        circuit_list: list[quartz.PyGraph | None] = [
            self.input_circuit
        ] * self.num_workers
        trajectory_len_list: list[int] = [0] * self.num_workers
        while True:
            expansion_cnt += 1
            if expansion_cnt % 100 == 0:
                print(
                    f'Expansion cnt: {expansion_cnt}, num sample circ: {len(self.circ_sample_set)}, none count: {self.none_count}, t: {time.time() - start:.2f}'
                )
                # print(self.visit_cnt)
            with torch.no_grad():
                circuit_list = self.expand(circuit_list)
            visit_cnt_reversed = [1 / self.visit_cnt[circ] for circ in self.circ_list]

            for i, circuit in enumerate(circuit_list):
                trajectory_len_list[i] += 1
                if (
                    circuit is not None
                    and trajectory_len_list[i] > self.max_trajectory_len
                ):
                    circuit = None

                if circuit is None:
                    circuit_list[i] = random.choices(
                        self.circ_list, visit_cnt_reversed, k=1
                    )[0]
                    self.visit_cnt[circuit_list[i]] += 1
                    self.none_count += 1
                    trajectory_len_list[i] = 0


@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    os.chdir(hydra.utils.get_original_cwd())  # set working dir to the original one

    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')

    # Build quartz context
    qtz: quartz.QuartzContext = quartz.QuartzContext(
        gate_set=['h', 'cx', 'x', 'rz', 'add'], filename='../ecc_set/nam_ecc.json'
    )

    # Device
    device = torch.device("cuda:1")

    # circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
    #     context=qtz, filename="../nam_circs/barenco_tof_10.qasm")
    circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
        context=qtz, filename="best_graphs/barenco_tof_3_cost_38.qasm"
    )

    # Load actor-critic network
    ckpt_path = "ckpts/iter_133.pt"
    actor_critic: ActorCritic = ActorCritic(
        gnn_type=cfg.gnn_type,
        num_gate_types=cfg.num_gate_types,
        gate_type_embed_dim=cfg.gate_type_embed_dim,
        gnn_num_layers=cfg.gnn_num_layers,
        gnn_hidden_dim=cfg.gnn_hidden_dim,
        gnn_output_dim=cfg.gnn_output_dim,
        gin_num_mlp_layers=cfg.gin_num_mlp_layers,
        gin_learn_eps=cfg.gin_learn_eps,
        gin_neighbor_pooling_type=cfg.gin_neighbor_pooling_type,
        gin_graph_pooling_type=cfg.gin_graph_pooling_type,
        actor_hidden_size=cfg.actor_hidden_size,
        critic_hidden_size=cfg.critic_hidden_size,
        action_dim=qtz.num_xfers,
        device=device,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_state_dict = cast(OrderedDict[str, torch.Tensor], ckpt['model_state_dict'])
    actor_critic.load_state_dict(model_state_dict)

    parallel_search_agent = ParallelSearchAgent(
        device=device,
        actor_critic=actor_critic,
        num_workers=64,
        hit_rate=0.95,
        max_cost_ratio=1.2,
        max_cost_increase=6,
        max_trajectory_len=600,
        qtz=qtz,
        circuit=circ,
    )

    parallel_search_agent.run()


if __name__ == "__main__":
    main()
