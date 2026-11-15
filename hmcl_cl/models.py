"""Encoder (reduced ResNet-18, trained from scratch) and Euclidean / hyperbolic heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as L
from .geometry import DynamicCurvature, FullyHyperbolicLinear


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.short = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        return F.relu(out + self.short(x))


class ConvEncoder(nn.Module):
    """Reduced ResNet-18 (one block per stage), trained from scratch.

    GroupNorm instead of BatchNorm so that BN running statistics (which no method can
    protect) do not add geometry-independent forgetting.
    """

    def __init__(self, embed_dim: int = 64, width: int = 64) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.GroupNorm(8, w), nn.ReLU())
        self.layers = nn.Sequential(
            BasicBlock(w, w, 1), BasicBlock(w, 2 * w, 2), BasicBlock(2 * w, 4 * w, 2), BasicBlock(4 * w, 8 * w, 2))
        self.proj = nn.Linear(8 * w, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = self.layers(self.stem(x))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.norm(self.proj(h))


class ContinualLearningModel(nn.Module):
    def __init__(self, mode: str, embed_dim: int = 64, width: int = 64,
                 curv_init: float = 0.1, curv_min: float = 0.05, curv_max: float = 0.5) -> None:
        super().__init__()
        self.mode = mode
        self.is_euclidean = mode in ("euclidean", "ewc", "ablation-euclidean")
        self.is_dynamic = mode == "dynamic-curvature"
        self.encoder = ConvEncoder(embed_dim, width)
        if self.is_euclidean:
            self.head = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.orthogonal_(self.head.weight)
        else:
            self.head = FullyHyperbolicLinear(embed_dim, curv=curv_init)
            self.curv_mod = DynamicCurvature(curv_init, curv_min, curv_max, learnable=self.is_dynamic)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.1).log())
        self.use_amp = False

    @property
    def curv(self) -> torch.Tensor:
        if self.is_euclidean:
            return torch.tensor(0.0)
        return self.curv_mod()

    def head_weight(self) -> torch.Tensor:
        return self.head.linear.weight if hasattr(self.head, "linear") else self.head.weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_amp and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.encoder(x.contiguous(memory_format=torch.channels_last))
            h = h.float()
        else:
            h = self.encoder(x)
        if self.is_euclidean:
            return F.normalize(self.head(h), p=2, dim=-1)
        return self.head(h, curv=self.curv)

    def logits(self, z: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        if self.is_euclidean:
            return (z @ protos.T) * self.logit_scale.exp()
        return -L.pairwise_dist(z, protos, self.curv) * self.logit_scale.exp()

    def set_task_start(self) -> None:
        if not self.is_euclidean:
            self.curv_mod.set_task_start()
