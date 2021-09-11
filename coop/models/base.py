import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 latent_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor = None,
                do_generate: torch.Tensor = False,
                **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def generate(self,
                 z: torch.Tensor,
                 num_beams: int = 4,
                 max_tokens: int = 256):
        raise NotImplementedError()

    @staticmethod
    def klw(step: int,
            interval: int,
            r: float = 0.8,
            t: float = 0.0,
            s: int = 10000):
        raise NotImplementedError()
