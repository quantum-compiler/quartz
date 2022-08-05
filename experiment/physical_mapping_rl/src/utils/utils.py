from quartz import PyAction, PyGraphState
import numpy as np
import math
import torch
import dgl


def py_action_2_list(action_list: [PyAction]):
    decoded_actions = []
    for action in action_list:
        decoded_actions.append((action.qubit_idx_0, action.qubit_idx_1))
    return decoded_actions


def action_2_id(input_action, action_space):
    for idx, action in enumerate(action_space):
        if action[0] == input_action[0] and action[1] == input_action[1]:
            return torch.tensor([idx])
    assert False


def graph_state_2_dgl(graph_state: PyGraphState):
    g = dgl.graph((torch.tensor(graph_state.edge_from, dtype=torch.int32),
                   torch.tensor(graph_state.edge_to, dtype=torch.int32)))
    g.edata['logical_idx'] = torch.tensor(graph_state.edge_logical_idx, dtype=torch.int32)
    g.edata['physical_idx'] = torch.tensor(graph_state.edge_physical_idx, dtype=torch.int32)
    g.edata['reversed'] = torch.tensor(graph_state.edge_reversed, dtype=torch.int32)
    g.ndata['is_input'] = torch.tensor(graph_state.is_input, dtype=torch.int32)
    return g


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    return e**2/2


def check(_input):
    output = torch.from_numpy(_input) if type(_input) == np.ndarray else _input
    return output
