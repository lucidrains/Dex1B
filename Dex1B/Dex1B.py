import torch
from torch.nn import Module, ModuleList

from x_transformers import Encoder

from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class VAE(Module):
    def __init__(self):
        super().__init__()


class Dex1B(Module):
    def __init__(self):
        super().__init__()
