import math
import os
import time
import warnings
from typing import OrderedDict, cast

import dgl
import hydra
import torch
import torch.nn.functional as F
from model.actor_critic import ActorCritic
from utils import masked_softmax

import quartz
from config.config import *


class Node:
    def __init__(self, circuit: quartz.PyGraph) -> None:
        self.circuit: quartz.PyGraph = circuit
        self.gate_count: int = self.circuit.gate_count
        self.g_xfer: list[tuple[int, int]] = []


class GreedySearchAgent:
    def __init__(
        self,
        qtz: quartz.QuartzContext,
        circuit: quartz.PyGraph,
        gate_count_increase: int,
        actor_critic: ActorCritic,
        device: torch.device,
    ) -> None:
        self.qtz: quartz.QuartzContext = qtz
        self.input_circuit: quartz.PyGraph = circuit
        self.init_gate_count: int = circuit.gate_count
        self.actor_critic: ActorCritic = actor_critic
        self.actor_critic.eval()
        self.device: torch.device = device

        self.node_list: list[Node] = [Node(circuit=circuit)]
        self.circ_set: set[quartz.PyGraph] = set()
        self.circ_set.add(circuit)
        self.min_gate_count: int = self.init_gate_count
        self.gate_count_increase: int = gate_count_increase

    def expand(self) -> bool:
        # Initialize node attributes
        node = self.node_list[-1]
        g_xfer: list[tuple[int, int]] = []
        P_: list[tuple[float, tuple[int, int]]] = []

        mask: torch.Tensor = torch.zeros(
            (node.gate_count, self.qtz.num_xfers), dtype=torch.bool
        )
        for g in range(node.gate_count):
            appliable_xfers: list[int] = node.circuit.available_xfers_parallel(
                context=self.qtz, node=node.circuit.get_node_from_id(id=g)
            )
            appliable_xfers = appliable_xfers[:-1]  # remove NOP
            mask[g, appliable_xfers] = True
            # construct child nodes
            for xfer in appliable_xfers:
                g_xfer.append((g, xfer))

        # Compute policy
        # Policy should take into consideration of gate values
        policy_table: torch.Tensor = torch.zeros((node.gate_count, self.qtz.num_xfers))
        with torch.no_grad():
            dgl_graph: dgl.DGLGraph = node.circuit.to_dgl_graph().to(self.device)
            node_embeds: torch.Tensor = self.actor_critic.gnn(dgl_graph)
            node_values: torch.Tensor = self.actor_critic.critic(node_embeds).squeeze()
            softmax_node_values = F.softmax(node_values, dim=0)
            xfer_logits: torch.Tensor = self.actor_critic.actor(node_embeds)
            xfer_probs: torch.Tensor = masked_softmax(xfer_logits, mask)
            for g, (node_prob, xfer_prob_list) in enumerate(
                zip(softmax_node_values, xfer_probs)
            ):
                policy_table[g] = node_prob * xfer_prob_list

        # Construct node.g_xfer
        for g, xfer in g_xfer:
            P_.append((policy_table[g, xfer].item(), (g, xfer)))
        P_.sort()
        for _, (g, xfer) in P_:
            node.g_xfer.append((g, xfer))

        success: bool = False
        while len(node.g_xfer) > 0:
            g, xfer = node.g_xfer.pop()
            py_xfer: quartz.PyXfer = self.qtz.get_xfer_from_id(id=xfer)
            if (
                node.gate_count - py_xfer.src_gate_count + py_xfer.dst_gate_count
                > self.min_gate_count + self.gate_count_increase
            ):
                continue
            child_circuit: quartz.PyGraph = node.circuit.apply_xfer(
                xfer=self.qtz.get_xfer_from_id(id=xfer),
                node=node.circuit.get_node_from_id(id=g),
                eliminate_rotation=True,
            )
            # Eliminate circuits that has been seen
            if child_circuit in self.circ_set:
                continue

            self.circ_set.add(child_circuit)
            self.node_list.append(Node(circuit=child_circuit))
            success = True
            if child_circuit.gate_count < self.min_gate_count:
                self.min_gate_count = child_circuit.gate_count
                print(f"min gate count: {self.min_gate_count}")
            break

        return success

    def back_track(self) -> None:
        terminated: bool = True
        while terminated:
            self.node_list.pop()
            node = self.node_list[-1]
            success: bool = False
            while len(node.g_xfer) > 0:
                g, xfer = node.g_xfer.pop()
                py_xfer: quartz.PyXfer = self.qtz.get_xfer_from_id(id=xfer)
                if (
                    node.gate_count - py_xfer.src_gate_count + py_xfer.dst_gate_count
                    > self.min_gate_count + self.gate_count_increase
                ):
                    continue
                child_circuit: quartz.PyGraph = node.circuit.apply_xfer(
                    xfer=self.qtz.get_xfer_from_id(id=xfer),
                    node=node.circuit.get_node_from_id(id=g),
                    eliminate_rotation=True,
                )
                # Eliminate circuits that has been seen
                if child_circuit in self.circ_set:
                    continue

                self.circ_set.add(child_circuit)
                self.node_list.append(Node(circuit=child_circuit))
                success = True
                if child_circuit.gate_count < self.min_gate_count:
                    self.min_gate_count = child_circuit.gate_count
                    print(f"min gate count: {self.min_gate_count}")
                return

            if not success:
                continue

    def run(self):
        # TODO: use the budget to stop the search
        budget = None
        expansion_cnt: int = 0
        start = time.time()

        while True:
            expansion_cnt += 1
            if expansion_cnt % 100 == 0:
                print(
                    f'Expansion cnt: {expansion_cnt}, num circuits: {len(self.circ_set)}, path len: {len(self.node_list)}, time: {time.time() - start}'
                )
            succeed: bool = self.expand()
            if not succeed:
                self.back_track()


@hydra.main(config_path='config', config_name='config')
def main(config: Config) -> None:
    output_dir = os.path.abspath(os.curdir)  # get hydra output dir
    os.chdir(hydra.utils.get_original_cwd())  # set working dir to the original one

    cfg: BaseConfig = config.c
    warnings.simplefilter('ignore')

    # Build quartz context
    qtz: quartz.QuartzContext = quartz.QuartzContext(
        gate_set=['h', 'cx', 'x', 'rz', 'add'], filename='../ecc_set/nam_ecc.json'
    )

    # Device
    device = torch.device("cuda:0")

    # Use the best circuit found in the PPO training
    # circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
    #     context=qtz, filename="../nam_circs/adder_8.qasm")
    circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
        context=qtz, filename="best_graphs/barenco_tof_3_cost_58.qasm"
    )

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
    model_state_dict = cast(OrderedDict[str, torch.Tensor], ckpt['model_state_dict'])
    actor_critic.load_state_dict(model_state_dict)

    # Initialize MCTS and run
    # TODO: tune the hyperparameters
    greedy_search_agent = GreedySearchAgent(
        device=device,
        actor_critic=actor_critic,
        qtz=qtz,
        gate_count_increase=2,
        circuit=circ,
    )

    greedy_search_agent.run()


if __name__ == "__main__":
    # os.environ['OMP_NUM_THREADS'] = str(16)
    main()
