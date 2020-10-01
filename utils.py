import torch
import variable as var
import numpy as np
from functools import reduce
import operator


def to_tensor(a):
    return torch.tensor(a, device=var.device).float()


def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)


def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value


def rolling_window(a, window=3):

    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad, mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)





