from __future__ import annotations

import torch
from torch import nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
