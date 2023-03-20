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

# TODO: De-duplicate the circuits
# - A circuit should appear only once in the tree. This is because
# the meaning of different appearances of the same circuit are
# exactly the same
# TODO: Prevent selecting invalid child: use mask?


class Node:
    def __init__(self, circuit: quartz.PyGraph) -> None:
        self.circuit: quartz.PyGraph = circuit
        self.gate_count: int = self.circuit.gate_count
        self.is_leaf: bool = True
        self.is_terminal: bool = False
        # Should be equal to the sum of child visit counts
        self.total_visit_count: int = 0

        # Child states and statics
        self.terminal_mask: torch.Tensor
        self.child_nodes: list[Node]
        self.Q: torch.Tensor
        self.N: torch.Tensor
        self.R: torch.Tensor
        self.P: torch.Tensor


class MCTSAgent:
    def __init__(
        self,
        gamma: float,
        c_1: float,
        c_2: float,
        qtz: quartz.QuartzContext,
        circuit: quartz.PyGraph,
        actor_critic: ActorCritic,
        max_cost_increase: int,
        hit_rate: float,
        device: torch.device,
    ) -> None:
        self.gamma: float = gamma
        self.c_0 = 0.0001  # debug use
        self.c_1: float = c_1
        self.c_2: float = c_2
        self.qtz: quartz.QuartzContext = qtz
        self.input_circuit: quartz.PyGraph = circuit
        self.init_cost: int = circuit.gate_count
        self.max_cost_increase: int = max_cost_increase
        self.hit_rate: float = hit_rate
        self.actor_critic: ActorCritic = actor_critic
        self.actor_critic.eval()
        self.device: torch.device = device

        self.root: Node = Node(circuit=circuit)
        self.circ_set: set[quartz.PyGraph] = set()
        self.circ_set.add(circuit)
        self.min_cost: int = self.init_cost
        self.longest_path: int = 0

    def select_child(self, node: Node) -> tuple[Node, int]:
        # Naive UCB
        # for child_node, value in zip(node.child_nodes, node.Q):
        #     print(child_node.gate_count, value)
        # ucb_scores: torch.Tensor = node.P + self.c_0 * torch.sqrt(
        #     math.log(node.total_visit_count) /
        #     (1 + node.N)) - node.terminal_mask * 1e10
        ucb_scores: torch.Tensor = (
            node.Q
            + node.P
            * math.sqrt(node.total_visit_count)
            / (1 + node.N)
            * (self.c_1 + math.log((node.total_visit_count + self.c_2 + 1) / self.c_2))
            - node.terminal_mask * 1e10
        )
        # print(node.Q)
        # print(node.P * math.sqrt(node.total_visit_count) / (1 + node.N) *
        #       (self.c_1 + math.log(
        #           (node.total_visit_count + self.c_2 + 1) / self.c_2)))
        # print(node.N)
        # Muzero UCB
        # ucb_scores: torch.Tensor = node.P * math.sqrt(
        #     node.total_visit_count) / (1 + node.N) * (self.c_1 + math.log(
        #         (node.total_visit_count + self.c_2 + 1) /
        #         self.c_2)) - node.terminal_mask * 1e10
        # print(ucb_scores)
        # print(ucb_scores.max())
        idx: int = torch.argmax(ucb_scores).item()
        # print(node.child_nodes[idx].gate_count, ucb_scores[idx])
        return node.child_nodes[idx], idx

    def select(self) -> tuple[list[Node], list[tuple[int, int]]]:
        # Returns a sequence of Node for backpropagate
        node_sequence: list[Node] = []
        path: list[int] = []
        curr_node: Node = self.root
        while not curr_node.is_leaf:
            node_sequence.append(curr_node)
            curr_node, idx = self.select_child(curr_node)
            path.append(idx)
        node_sequence.append(curr_node)
        return node_sequence, path

    def expand(self, node: Node) -> bool:
        # Returns False if the node has no legal child

        # Initialize node attributes
        node.child_nodes = []
        R_: list[torch.Tensor] = []
        P_: list[torch.Tensor] = []

        # Compute mask
        mask: torch.Tensor = torch.zeros(
            (node.gate_count, self.qtz.num_xfers), dtype=torch.bool
        )
        for g in range(node.gate_count):
            appliable_xfers: list[int] = node.circuit.available_xfers_parallel(
                context=self.qtz, node=node.circuit.get_node_from_id(id=g)
            )
            # appliable_xfers = appliable_xfers[:-1]  # remove NOP
            mask[g, appliable_xfers] = True

        # Compute policy
        # Policy here is the probability of selecting a node (or gate)
        # In evaluation, we use the argmax of the xfer policy
        dgl_graph: dgl.DGLGraph = node.circuit.to_dgl_graph().to(self.device)
        node_embeds: torch.Tensor = self.actor_critic.gnn(dgl_graph)
        node_values: torch.Tensor = self.actor_critic.critic(node_embeds).squeeze()
        temperatures = 1 / (
            math.log(self.hit_rate * (node.gate_count - 1) / (1 - self.hit_rate))
        )
        softmax_node_values = F.softmax(node_values / temperatures, dim=0)

        # Decide which xfer to take at each node
        xfer_logits: torch.Tensor = self.actor_critic.actor(node_embeds)
        xfer_probs: torch.Tensor = masked_softmax(xfer_logits, mask)
        xfers: list[int] = torch.argmax(xfer_probs, dim=1).tolist()

        # Construct child nodes
        for g, x in enumerate(xfers):
            xfer: quartz.PyXfer = self.qtz.get_xfer_from_id(id=x)
            if xfer.is_nop:
                continue
            else:
                reward: int = xfer.src_gate_count - xfer.dst_gate_count
                if node.gate_count - reward > self.min_cost + self.max_cost_increase:
                    continue
                new_circ: quartz.PyGraph = node.circuit.apply_xfer(
                    xfer=xfer,
                    node=node.circuit.get_node_from_id(id=g),
                    eliminate_rotation=True,
                )
                if new_circ.gate_count < self.min_cost:
                    print("New min cost: ", new_circ.gate_count)
                    exit(1)
                # Eliminate duplication
                if new_circ in self.circ_set:
                    continue
                self.circ_set.add(new_circ)

                # Construct new node
                node.child_nodes.append(Node(circuit=new_circ))
                R_.append(reward)
                P_.append(softmax_node_values[g])

        # No child, a terminal node
        if len(R_) == 0:
            node.is_terminal = True
            node.is_leaf = False
            return False

        node.R = torch.Tensor(R_).to(self.device)
        node.P = torch.stack(P_)
        node.terminal_mask = torch.zeros(node.R.shape, dtype=torch.bool).to(self.device)
        # print(f"Expanding node with {len(node.child_nodes)} children")
        # Modify leaf flag
        node.is_leaf = False
        return True

    def simulate(self, node: Node) -> tuple[int, torch.Tensor]:
        # Instead of running a rollout
        # we just use the sum of gate values as the node value
        # In this method, we visit all of the child nodes once
        Q_: list[torch.Tensor] = []
        child_visit_count: int = len(node.child_nodes)
        total_value: torch.Tensor = 0
        b_dgl_graph = dgl.batch(
            [n.circuit.to_dgl_graph().to(self.device) for n in node.child_nodes]
        )
        num_nodes: list[int] = [n.gate_count for n in node.child_nodes]
        b_node_embeds: torch.Tensor = self.actor_critic.gnn(b_dgl_graph)
        b_values: torch.Tensor = self.actor_critic.critic(b_node_embeds).squeeze()
        values: list[torch.Tensor] = torch.split(b_values, num_nodes)
        for n, v in zip(node.child_nodes, values):
            Q_.append(torch.sum(v) - (n.gate_count - self.init_cost) * 10)

        node.Q = torch.stack(Q_)
        node.N = torch.ones(node.Q.shape, dtype=torch.int32).to(self.device)
        total_value = torch.sum((node.Q + node.R) * node.P) * child_visit_count
        node.total_visit_count = child_visit_count
        return child_visit_count, total_value

    def backpropagate(
        self,
        node_sequence: list[Node],
        path: list[int],
        value: torch.Tensor,
        visit_count: int,
    ) -> None:
        for node, idx in zip(reversed(node_sequence[:-1]), reversed(path)):
            node.Q[idx] = (node.Q[idx] * node.N[idx] + value) / (
                node.N[idx] + visit_count
            )
            # print(
            #     f"gate count: {node.gate_count}, visit count: {node.total_visit_count}, total nodes: {len(self.circ_set)}"
            # )
            node.N[idx] += visit_count
            node.total_visit_count += visit_count
            value = value * self.gamma + node.R[idx]
        # print(f"total nodes: {len(self.circ_set)}")

    def run(self):
        with torch.no_grad():
            expansion_cnt: int = 0
            start = time.time()
            while True:
                expansion_cnt += 1
                if expansion_cnt % 100 == 0:
                    print(
                        f'Expansion count: {expansion_cnt}, num circuits: {len(self.circ_set)}, longest path: {self.longest_path}, time: {time.time() - start}'
                    )
                node_sequence, path = self.select()
                if len(path) > self.longest_path:
                    self.longest_path = len(path)
                not_terminated: bool = self.expand(node_sequence[-1])
                if not not_terminated:
                    for node, idx in zip(reversed(node_sequence[:-1]), reversed(path)):
                        node.terminal_mask[idx] = True
                        node.child_nodes[idx] = None
                        if not torch.all(node.terminal_mask):
                            break
                    continue
                visit_count, value = self.simulate(node_sequence[-1])
                self.backpropagate(node_sequence, path, value, visit_count)


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
    circ: quartz.PyGraph = quartz.PyGraph().from_qasm(
        context=qtz, filename="best_graphs/barenco_tof_3_cost_38.qasm"
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
    mcts = MCTSAgent(
        gamma=1.00,
        c_1=32,
        c_2=19625,
        device=device,
        actor_critic=actor_critic,
        max_cost_increase=6,
        hit_rate=0.95,
        qtz=qtz,
        circuit=circ,
    )

    mcts.run()


if __name__ == "__main__":
    main()
