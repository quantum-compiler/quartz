import torch
import quartz
import dgl
from model.actor_critic import ActorCritic
from utils import masked_softmax
import torch.nn.functional as F
import math
from config.config import *
import hydra
import os
import warnings
from typing import cast, OrderedDict


class Node:
    def __init__(self, circuit: quartz.PyGraph, num_xfers: int) -> None:
        self.circuit: quartz.PyGraph = circuit
        self.gate_count: int = self.circuit.gate_count
        self.is_leaf: bool = True
        # Should be equal to the sum of child visit counts
        self.total_visit_count: int = 0

        # Child states and statics
        self.child_nodes: list[list[Node | None]]
        self.action_mask: torch.Tensor
        self.Q: torch.Tensor
        self.N: torch.Tensor
        self.R: torch.Tensor
        self.P: torch.Tensor
        # Initialize
        self.child_nodes = [[None for _ in range(num_xfers)]
                            for i in range(self.gate_count)]
        self.N = torch.zeros((self.gate_count, num_xfers), dtype=torch.int32)
        self.R = torch.zeros((self.gate_count, num_xfers), dtype=torch.float32)
        self.Q = torch.zeros((self.gate_count, num_xfers), dtype=torch.float32)
        self.P = torch.zeros((self.gate_count, num_xfers), dtype=torch.float32)


class MCTSAgent:
    def __init__(
        self,
        gamma: float,
        c_1: float,
        c_2: float,
        quartz_context: quartz.QuartzContext,
        circuit: quartz.PyGraph,
        actor_critic: ActorCritic,
        device: torch.device,
    ) -> None:
        self.gamma: float = gamma
        self.c_1: float = c_1
        self.c_2: float = c_2
        self.quartz_context: quartz.QuartzContext = quartz_context
        self.input_circuit: quartz.PyGraph = circuit
        self.init_gate_count: int = circuit.gate_count
        self.actor_critic: ActorCritic = actor_critic
        self.actor_critic.eval()
        self.device: torch.device = device

        self.root: Node = Node(circuit=circuit,
                               num_xfers=quartz_context.num_xfers)

    def select_child(self, node: Node) -> tuple[Node, int, int]:
        ucb_scores: torch.Tensor = node.Q + node.P * math.sqrt(
            node.total_visit_count) / (1 + node.N) * (self.c_1 + math.log(
                (node.total_visit_count + self.c_2 + 1) / self.c_2))
        idx: int = torch.argmax(ucb_scores).item()
        g: int = idx // self.quartz_context.num_xfers
        xfer: int = idx % self.quartz_context.num_xfers
        return node.child_nodes[g][xfer], g, xfer

    def select(self) -> tuple[list[Node], list[tuple[int, int]]]:
        # Returns a sequence of Node for backpropagate
        node_sequence: list[Node] = []
        path: list[tuple[int, int]] = []
        curr_node: Node = self.root
        while not curr_node.is_leaf:
            node_sequence.append(curr_node)
            curr_node, g, xfer = self.select_child(curr_node)
            path.append((g, xfer))
        node_sequence.append(curr_node)
        return node_sequence, path

    def expand(self, node: Node) -> None:
        # During expansion, we expand a node to get all of its child nodes
        # we construct a mask to indicate which actions are appliable
        # for the appliable actions, there is a reward
        # N and Q corresponding to a child node is initialized to 0
        # we also compute the policy
        # which is, basically, a probability distribution over all the child nodes
        masks: list[torch.Tensor] = []
        for g in range(node.gate_count):
            appliable_xfers: list[int] = node.circuit.available_xfers_parallel(
                context=self.quartz_context,
                node=node.circuit.get_node_from_id(id=g))
            # construct mask
            mask: torch.Tensor = torch.zeros((self.quartz_context.num_xfers),
                                             dtype=torch.bool)
            mask[appliable_xfers] = True
            mask[-1] = False  # exclude NOP
            masks.append(mask)
            # construct child nodes
            for xfer in appliable_xfers:
                child_circuit: quartz.PyGraph = node.circuit.apply_xfer(
                    xfer=self.quartz_context.get_xfer_from_id(id=xfer),
                    node=node.circuit.get_node_from_id(id=g))
                # TODO: Eliminate circuits that has been seen?
                node.child_nodes[g][xfer] = Node(child_circuit,
                                                 self.quartz_context.num_xfers)
                # Fill in R
                node.R[g][xfer] = node.gate_count - child_circuit.gate_count
        node.action_mask = torch.stack(masks, dim=0)

        # Compute policy
        # Policy should take into consideration of gate values
        dgl_graph: dgl.DGLGraph = node.circuit.to_dgl_graph().to(self.device)
        node_embeds: torch.Tensor = self.actor_critic.gnn(dgl_graph)
        node_values: torch.Tensor = self.actor_critic.critic(
            node_embeds).squeeze().cpu()
        softmax_node_values = F.softmax(node_values, dim=0)
        xfer_logits: torch.Tensor = self.actor_critic.actor(node_embeds).cpu()
        xfer_probs: torch.Tensor = masked_softmax(xfer_logits,
                                                  node.action_mask)
        for g, (node_prob, xfer_prob_list) in enumerate(
                zip(softmax_node_values, xfer_probs)):
            node.P[g] = node_prob * xfer_prob_list

        # Modify leaf flag
        node.is_leaf = False

    def simulate(self, node: Node) -> tuple[int, torch.Tensor]:
        # Instead of running a rollout
        # we just use the sum of gate values as the node value
        # In this method, we visit all of the child nodes once
        child_visit_count: int = 0
        total_value: torch.Tensor = 0
        for g in range(node.gate_count):
            for xfer in range(self.quartz_context.num_xfers):
                if node.child_nodes[g][xfer] is not None:
                    with torch.no_grad():
                        dgl_graph: dgl.DGLGraph = node.child_nodes[g][
                            xfer].circuit.to_dgl_graph().to(self.device)
                        node_embeds: torch.Tensor = self.actor_critic.gnn(
                            dgl_graph)
                        # The node value is sum(gate_values) - gate_count + init_gate_count
                        node.Q[g][xfer] = torch.sum(
                            self.actor_critic.critic(node_embeds).squeeze().
                            cpu()) - node.child_nodes[g][
                                xfer].circuit.gate_count + self.init_gate_count
                        node.N[g][xfer] = 1
                        child_visit_count += 1
                        total_value += (node.Q[g][xfer] + node.R[g][xfer])
        node.total_visit_count = child_visit_count
        return child_visit_count, total_value

    def backpropagate(self, node_sequence: list[Node], path: list[tuple[int,
                                                                        int]],
                      value: torch.Tensor, visit_count: int) -> None:
        for node, (g, xfer) in zip(reversed(node_sequence[:-1]),
                                   reversed(path)):
            node.Q[g][xfer] = (node.Q[g][xfer] * node.N[g][xfer] +
                               value) / (node.N[g][xfer] + visit_count)
            node.N[g][xfer] += visit_count
            value = value * self.gamma + node.R[g][xfer]

    def run(self):
        # TODO: use the budget to stop the search
        budget = None
        expansion_cnt: int = 0
        while True:
            expansion_cnt += 1
            if expansion_cnt % 1000 == 0:
                print(f'Expansion count: {expansion_cnt}')
            node_sequence, path = self.select()
            self.expand(node_sequence[-1])
            visit_count, value = self.simulate(node_sequence[-1])
            print(visit_count, value)
            self.backpropagate(node_sequence, path, value, visit_count)
            # TODO: print more messages
            min_gate_count = min([node.gate_count for node in node_sequence])
            print(min_gate_count)


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
    device = torch.device("cuda:0")

    # TODO: Build circuit
    # use the best circuit found in the PPO training
    circ: quartz.PyGraph = quartz.PyGraph().from_qasm(context=qtz,
                                                      filename="test.qasm")

    # TODO: load actor-critic network
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

    # TODO: Initialize MCTS and run
    mcts = MCTSAgent(gamma=0.99,
                     c_1=1.25,
                     c_2=19625,
                     device=device,
                     actor_critic=actor_critic,
                     quartz_context=qtz,
                     circuit=circ)

    mcts.run()


if __name__ == "__main__":
    main()