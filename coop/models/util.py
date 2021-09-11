from typing import NamedTuple

import torch
from torch.distributions import Normal


class Losses(NamedTuple):
    nll: torch.Tensor
    zkl: torch.Tensor
    zkl_real: torch.Tensor


class VAEOut(NamedTuple):
    q: Normal
    generated: torch.Tensor = None
