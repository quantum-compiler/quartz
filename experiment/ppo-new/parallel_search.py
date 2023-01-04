import torch
import quartz
import dgl
from model.actor_critic import ActorCritic
from utils import masked_softmax
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from config.config import *
import hydra
import os
import warnings
from typing import cast, OrderedDict
import time
import random


class ParallelSearchAgent:
    def __init__(
        self,
        qtz: quartz.QuartzContext,
        circuit: quartz.PyGraph,
        num_workers: int,
        gate_count_increase: int,
        max_cost_ratio: float,
        actor_critic: ActorCritic,
        hit_rate: float,
        device: torch.device,
    ) -> None:
        self.qtz: quartz.QuartzContext = qtz
        self.input_circuit: quartz.PyGraph = circuit
        self.init_gate_count: int = circuit.gate_count
        self.min_gate_count: int = self.init_gate_count
        self.max_cost_ratio: float = max_cost_ratio
        self.actor_critic: ActorCritic = actor_critic
        self.actor_critic.eval()
        self.num_workers: int = num_workers
        self.gate_count_increase: int = gate_count_increase
        self.hit_rate: float = hit_rate
        self.device: torch.device = device

        self.circ_set: set[quartz.PyGraph] = set()
        self.circ_set.add(circuit)
        self.circ_buffer: dict[int, set[quartz.PyGraph]] = {}
        self.circ_buffer[self.init_gate_count] = set([circuit])
        self.circ_sample_set: set[quartz.PyGraph] = set()
        self.circ_sample_set.add(circuit)

        self.none_count: int = 0

    def expand(
            self,
            circuit_list: list[quartz.PyGraph]) -> list[quartz.PyGraph | None]:
        b_circs: dgl.DGLGraph = dgl.batch([
            circuit.to_dgl_graph() for circuit in circuit_list
        ]).to(self.device)
        num_nodes: torch.Tensor = b_circs.batch_num_nodes()
        b_node_embeds: torch.Tensor = self.actor_critic.gnn(b_circs)
        b_node_values: torch.Tensor = self.actor_critic.critic(
            b_node_embeds).squeeze()
        node_embeds_list: list[torch.Tensor] = torch.split(
            b_node_embeds, num_nodes.tolist())

        # Sample nodes
        node_values_list: list[torch.Tensor] = torch.split(
            b_node_values, num_nodes.tolist())
        temperatures = 1 / (torch.log(self.hit_rate * (num_nodes - 1) /
                                      (1 - self.hit_rate)))
        nodes: list[int] = [
            torch.multinomial(F.softmax(node_values / temperature, dim=0),
                              1)[0].item()
            for node_values, temperature in zip(node_values_list, temperatures)
        ]

        # Construct masks
        mask: torch.Tensor = torch.zeros(
            (self.num_workers, self.qtz.num_xfers), dtype=torch.bool)
        for i, (n, circ) in enumerate(zip(nodes, circuit_list)):
            appliable_xfers: list[int] = circ.available_xfers_parallel(
                context=self.qtz, node=circ.get_node_from_id(id=n))
            mask[i][appliable_xfers] = True

        # Sample xfers
        selected_node_embeds: list[torch.Tensor] = [
            node_embeds[node]
            for node_embeds, node in zip(node_embeds_list, nodes)
        ]
        b_selected_node_embeds: torch.Tensor = torch.stack(
            selected_node_embeds, dim=0)
        b_xfer_logits: torch.Tensor = self.actor_critic.actor(
            b_selected_node_embeds)
        b_xfer_probs: torch.Tensor = masked_softmax(b_xfer_logits, mask)
        dist: Categorical = Categorical(b_xfer_probs)
        xfers: list[int] = dist.sample().tolist()

        # print(f'Nodes: {nodes}')
        # print(f'Xfers: {xfers}')

        # Construct new circuits
        new_circ_list: list[quartz.PyGraph] = []
        for i, (n, x, circ) in enumerate(zip(nodes, xfers, circuit_list)):
            xfer: quartz.PyXfer = self.qtz.get_xfer_from_id(id=x)
            if xfer.is_nop:
                new_circ_list.append(None)
                self.none_count += 1
            # elif circ.gate_count + xfer.dst_gate_count - xfer.src_gate_count > self.min_gate_count + self.gate_count_increase:
            #     new_circ_list.append(None)
            #     self.none_count += 1
            else:
                new_circ: quartz.PyGraph = circ.apply_xfer(
                    xfer=xfer,
                    node=circ.get_node_from_id(id=n),
                    eliminate_rotation=True)
                new_circ_list.append(new_circ)
                if new_circ not in self.circ_set:
                    self.circ_set.add(new_circ)

                    if new_circ.gate_count > self.min_gate_count + 1:
                        continue

                    if new_circ.gate_count not in self.circ_buffer:
                        self.circ_buffer[new_circ.gate_count] = set()
                    self.circ_buffer[new_circ.gate_count].add(new_circ)

                    if new_circ.gate_count < self.min_gate_count:
                        self.min_gate_count = new_circ.gate_count
                        print(f'New min gate count: {self.min_gate_count}')
                        self.circ_sample_set = set()
                        gate_count_to_pop: list[int] = []
                        for gate_count in self.circ_buffer:
                            # if gate_count > self.min_gate_count + self.gate_count_increase:
                            #     gate_count_to_pop.append(gate_count)
                            if gate_count > self.min_gate_count + 1:
                                gate_count_to_pop.append(gate_count)
                            else:
                                self.circ_sample_set.update(
                                    self.circ_buffer[gate_count])
                        for gate_count in gate_count_to_pop:
                            self.circ_buffer.pop(gate_count)
                    # else:
                    #     self.circ_sample_set.add(new_circ)
                    elif new_circ.gate_count <= self.min_gate_count + 1:
                        self.circ_sample_set.add(new_circ)

        return new_circ_list

    def run(self):
        start = time.time()
        expansion_cnt: int = 0
        circuit_list: list[quartz.PyGraph
                           | None] = [self.input_circuit] * self.num_workers
        init_gate_count_list: list[int] = [
            circuit.gate_count for circuit in circuit_list
        ]
        while True:
            expansion_cnt += 1
            if expansion_cnt % 100 == 0:
                print(
                    f'Expansion cnt: {expansion_cnt}, num circ: {len(self.circ_set)}, num sample circ: {len(self.circ_sample_set)}, none count: {self.none_count}, t: {time.time() - start:.2f}'
                )
            with torch.no_grad():
                circuit_list = self.expand(circuit_list)
            none_cnt: int = 0
            for i, circuit in enumerate(circuit_list):
                if circuit is None:
                    circuit_list[i] = random.sample(self.circ_sample_set, 1)[0]
                    init_gate_count_list[i] = circuit_list[i].gate_count
                    none_cnt += 1
                elif circuit_list[i].gate_count > init_gate_count_list[
                        i] * self.max_cost_ratio:
                    circuit_list[i] = random.sample(self.circ_sample_set, 1)[0]
                    init_gate_count_list[i] = circuit_list[i].gate_count
                    none_cnt += 1

            # print(f"new circuit count: {none_cnt}")


@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    output_dir = os.path.abspath(os.curdir)  # get hydra output dir
    os.chdir(
        hydra.utils.get_original_cwd())  # set working dir to the original one

    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')

    # Build quartz context
    qtz: quartz.QuartzContext = quartz.QuartzContext(
        gate_set=['h', 'cx', 'x', 'rz', 'add'],
        filename='../ecc_set/nam_ecc.json')

    # Device
    device = torch.device("cuda:3")

    # Use the best circuit found in the PPO training
    # circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
    #     context=qtz, filename="../nam_circs/adder_8.qasm")
    circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
        context=qtz, filename="best_graphs/barenco_tof_3_cost_38.qasm")

    # Load actor-critic network
    ckpt_path = "ckpts/nam_iter_100.pt"
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
    model_state_dict = cast(OrderedDict[str, torch.Tensor],
                            ckpt['model_state_dict'])
    actor_critic.load_state_dict(model_state_dict)

    # TODO: tune the hyperparameters
    parallel_search_agent = ParallelSearchAgent(device=device,
                                                actor_critic=actor_critic,
                                                num_workers=128,
                                                hit_rate=0.9,
                                                gate_count_increase=2,
                                                max_cost_ratio=1.2,
                                                qtz=qtz,
                                                circuit=circ)

    parallel_search_agent.run()


if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = str(16)
    main()