import quartz
from model import ActorCritic
import torch

# Global config
total_num_gates = 29
graph_embed_dim = 128
actor_hidden_dim = 256
critic_hidden_dim = 128
hit_rate = 0.9

gate_set = ['h', 'cx', 'x', 'rz', 'add']
ECC_fn = 'Nam_complete_ECC_set.json'
no_increase = False
include_nop = True

ckpt_path = "pretrained_model.pt"


class Node:
    def __init__(self, circ: quartz.PyGraph, node_id: int, parent_id: int):
        self.circ: quartz.PyGraph = circ
        self.visit_cnt: int = 0
        self.node_id: int = node_id
        self.parent_id: int = parent_id
        self.children_id: list[int] = []
        self.value: float = 0

    def is_root(self) -> bool:
        return self.node_id == 0

    def is_leaf(self) -> bool:
        return self.children_id == []

    def is_new(self) -> bool:
        return self.visit_cnt == 0

    def get_parent(self) -> int:
        return self.parent_id

    def get_children(self) -> list[int]:
        return self.children_id

    def visit(self) -> None:
        self.visit_cnt += 1

    def get_visit_cnt(self) -> int:
        return self.visit_cnt

    def get_value(self) -> float:
        return self.value

    def add_value(self, reward: float) -> None:
        self.value += reward

    def get_id(self) -> int:
        return self.node_id

    def get_circ(self) -> quartz.PyGraph:
        return self.circ

    def add_child(self, child_id: int) -> None:
        self.children_id.append(child_id)


class Nodes:
    def __init__(self):
        self.node_list: list[Node] = []
        self.node_cnt: int = 0

    def create_node(self, circ: quartz.PyGraph, parent_id: int) -> Node:
        id: int = self.node_cnt
        n: Node = Node(circ, id, parent_id)
        self.node_list.append(n)
        self.node_cnt += 1
        return n

    def get_node(self, node_id: int) -> Node:
        assert (node_id < self.node_cnt)
        return self.node_list[node_id]


class MCTS:
    def __init__(self, init_circ: quartz.PyGraph, C: float, k: int,
                 budget: int, max_step_length: int, gamma: float, device):
        self.context = quartz.QuartzContext(gate_set=gate_set,
                                            filename=ECC_fn,
                                            no_increase=no_increase,
                                            include_nop=include_nop)
        self.model = ActorCritic(total_num_gates, graph_embed_dim,
                                 actor_hidden_dim, critic_hidden_dim,
                                 self.context.num_xfers, hit_rate,
                                 device).to(device)
        self.model.load_ckpt(ckpt_path)
        self.Nodes: Nodes = Nodes()
        self.root: Node = self.Nodes.create_node(init_circ, 0)
        self.k: int = k
        self.C: float = C
        self.max_step_length = max_step_length
        self.gamma = gamma
        self.device = device

    def select_child(self, node: Node) -> Node:
        children_id_list = node.get_children()
        children_nodes: list[Node] = []
        children_visit_cnts: list[int] = []
        children_values: list[float] = []
        for child_id in children_id_list:
            child_node = self.Nodes.get_node(child_id)

            if child_node.get_visit_cnt() == 0:
                return child_node

            children_nodes.append(child_node)
            children_visit_cnts.append(child_node.get_visit_cnt())
            children_values.append(child_node.get_value())

        children_values: torch.Tensor = torch.tensor(children_values,
                                                     dytpe=torch.float)
        children_visit_cnts: torch.Tensor = torch.tensor(children_visit_cnts,
                                                         dtype=torch.float)
        visit_cnt: torch.Tensor = torch.tensor(node.get_visit_cnt(),
                                               dtype=torch.float)

        UCB: torch.Tensor = children_values + visit_cnt.log().div(
            children_visit_cnts).sqrt().mul(self.C)
        max_idx: int = UCB.argmax().item()

        return children_nodes[max_idx]

    def selection(self) -> tuple[Node, list[int]]:
        node_id_trace: list[int] = [0]
        node: Node = self.root
        while not node.is_leaf():
            # A visit is counted when a child of it is selected
            node.visit()
            node = self.select_child(node)
            node_id_trace.append(node.get_id())
        return node, node_id_trace

    def expansion(self, node: Node) -> Node:
        circ = node.get_circ()
        nodes, xfers = self.model.get_nodes_and_xfers_deterministic(
            self.context, circ, self.k)
        first_child: Node = None
        for n_id, x_id in zip(nodes, xfers):
            new_circ, _ = circ.apply_xfer_and_node_state_tracking(
                node=circ.get_node_from_id(id=n_id),
                xfer=self.context.get_xfer_from_id(id=x_id),
                eliminate_ratation=self.context.has_parameterized_gate())
            new_node = self.Nodes.create_node(new_circ, node.get_id())
            node.add_child(new_node.get_id())
            if first_child == None:
                first_child = new_node

        return first_child

    def simulation(self, node: Node) -> float:
        # TODO: run 1 trajectory from the node's circuit
        pass

    def backpropagation(self, reward: float, node_id_trace: list[int]) -> None:
        for node_id in reversed(node_id_trace):
            self.Nodes.get_node(node_id).add_value(reward)
            reward *= self.gamma

    def run(self):
        while self.budget > 0:
            node, node_id_trace = self.selection()

            # A visit is counted when a leaf is reached
            node.visit()

            if node.circ == None:
                continue

            self.budget -= 1
            if node.is_new():
                reward: float = self.simulation(node)
                self.backpropagation(reward, node_id_trace)
            else:
                child_node = self.expansion(node)
                reward: float = self.simulation(child_node)
                node_id_trace.append(child_node.get_id())
                self.backpropagationk(reward, node_id_trace)
