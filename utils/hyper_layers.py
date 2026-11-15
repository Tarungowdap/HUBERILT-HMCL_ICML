"""Neural layers required by HMCL."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import lorentz as L


class FullyHyperbolicLinear(nn.Module):
    """Bias-free Lorentz-space linear map used by the shared HMCL head."""

    def __init__(self, embed_dim: int, curv: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.register_buffer("curv", torch.tensor(float(curv)))
        self.linear = nn.Linear(embed_dim + 1, embed_dim + 1, bias=False)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, features: torch.Tensor, curv: float | torch.Tensor | None = None) -> torch.Tensor:
        effective_curv = curv if curv is not None else self.curv
        augmented = L.concat_time_dimension(features, effective_curv)
        transformed = self.linear(augmented)
        return transformed[..., 1:]
