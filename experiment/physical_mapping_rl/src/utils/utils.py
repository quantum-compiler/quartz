from quartz import PyAction
import numpy as np
import math
import torch


def DecodePyActionList(action_list: [PyAction]):
    decoded_actions = []
    for action in action_list:
        decoded_actions.append((action.qubit_idx_0, action.qubit_idx_0))
    return decoded_actions


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
