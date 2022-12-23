import torch
import quartz
import dgl
from model.actor_critic import ActorCritic
from utils import masked_softmax
import torch.functional as F
import math


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

    def check_leaf(self):
        # Check if the node is a leaf node
        # A node is a leaf node if not all of its child nodes are visited
        if torch.min(self.N[self.mask]) > 0:
            self.is_leaf = False
        self.is_leaf = True


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
        ucb_scores: torch.Tensor = self.Q + self.P * math.sqrt(
            self.total_visit_count) / (1 + self.N) * (self.c_1 + math.log(
                (self.total_visit_count + self.c_2 + 1) / self.c_2))
        idx: int = torch.argmax(ucb_scores).item()
        g: int = idx // self.quartz_context.num_xfers
        xfer: int = idx % self.quartz_context.num_xfers
        return node.child_nodes[g][xfer], g, xfer

    def select(self) -> tuple[list[Node], list[tuple[int, int]]]:
        # Returns a sequence of Node for backpropagate
        node_sequence: list[Node] = []
        path: list[tuple[int, int]] = []
        curr_node: Node = self.root
        # TODO: maybe we don't need check_leaf()?
        curr_node.check_leaf()
        while not curr_node.is_leaf:
            node_sequence.append(curr_node)
            curr_node, g, xfer = self.select_child(curr_node)
            path.append((g, xfer))
            curr_node.check_leaf()
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
                node.child_nodes[g][xfer] = Node(self.quartz_context,
                                                 child_circuit)
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
        xfer_logits: torch.Tensor = self.ac_net.actor(node_embeds).cpu()
        xfer_probs: torch.Tensor = masked_softmax(xfer_logits,
                                                  node.action_mask)
        for g, (node_prob, xfer_prob_list) in enumerate(
                zip(softmax_node_values, xfer_probs)):
            node.P[g] = node_prob * xfer_prob_list

    def simulate(self, node: Node) -> tuple[int, torch.Tensor]:
        # Instead of running a rollout
        # we just use the sum of gate values as the node value
        # In this method, we visit all of the child nodes once
        child_visit_count: int = 0
        total_value: torch.Tensor = 0
        for g in range(node.gate_count):
            for xfer in range(self.quartz_context.num_xfers):
                if node.child_nodes[g][xfer] is not None:
                    dgl_graph: dgl.DGLGraph = node.child_nodes[g][
                        xfer].circuit.to_dgl_graph().to(self.device)
                    node_embeds: torch.Tensor = self.actor_critic.gnn(
                        dgl_graph)
                    # The node value is sum(gate_values) - gate_count + init_gate_count
                    node.Q[g][xfer] = torch.sum(
                        self.actor_critic.critic(node_embeds).squeeze().cpu()
                    ) - node.child_nodes[g][
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
        while True:
            node_sequence, path: list[Node] = self.select()
            self.expand(node_sequence[-1])
            value, visit_count = self.simulate(node_sequence[-1])
            self.backpropagate(node_sequence, path, value, visit_count)
            # TODO: print some messages


if __name__ == "__main__":
    # TODO: load actor-critic network
    ckpt_path = "checkpoints/2021-06-01-15-30-00"

    # TODO: Build quartz context

    # TODO: Build circuit

    # TODO: Initialize MCTS and run
