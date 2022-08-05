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
