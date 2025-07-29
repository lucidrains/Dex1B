import torch
from torch.nn import Module, ModuleList

from x_transformer import Encoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class VAE(Module):
    def __init__(self):
        super().__init__()
