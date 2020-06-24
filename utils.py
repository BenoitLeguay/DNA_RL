import torch


def to_tensor(a, device='cuda:0'):
    return torch.tensor(a, device=device).float()
