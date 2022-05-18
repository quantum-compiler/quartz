import torch
import torch.nn.functional as F


def masked_softmax(logits, mask):
    mask = torch.ones_like(mask, dtype=torch.bool) ^ mask
    logits[mask] -= 1.0e+10
    return F.softmax(logits, dim=-1)