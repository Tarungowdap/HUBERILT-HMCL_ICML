"""Shared adaptation heads for HMCL, Dynamic-Curvature, and Euclidean baselines."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from utils.hyper_layers import FullyHyperbolicLinear


class DynamicCurvature(nn.Module):
    """Bounded, learnable Riemannian curvature parameterization.

    Constrains effective curvature c in [c_min, c_max] via sigmoid reparameterization
    and provides a soft task-start anchor penalty to prevent geometric collapse:
        L_anchor = 0.5 * reg * (c - c_start)^2.
    """

    def __init__(
        self,
        initial_c: float = 0.1,
        c_min: float = 0.05,
        c_max: float = 0.5,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.learnable = learnable

        clamped_init = max(self.c_min + 1e-4, min(self.c_max - 1e-4, float(initial_c)))
        norm_val = (clamped_init - self.c_min) / (self.c_max - self.c_min)
        init_raw = math.log(norm_val / (1.0 - norm_val))

        if learnable:
            self.raw_c = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        else:
            self.register_buffer("raw_c", torch.tensor(init_raw, dtype=torch.float32))

        self.register_buffer("task_start_c", torch.tensor(clamped_init, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return scalar effective curvature c strictly inside [c_min, c_max]."""
        sig = torch.sigmoid(self.raw_c)
        return self.c_min + (self.c_max - self.c_min) * sig

    def set_task_start(self) -> None:
        """Anchor the current curvature as the task reference."""
        with torch.no_grad():
            self.task_start_c.copy_(self.forward())

    def get_anchor_loss(self, reg_weight: float) -> torch.Tensor:
        """Calculate soft anchor loss: 0.5 * reg * (c - c_start)^2."""
        if not self.learnable or reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.raw_c.device)
        current_c = self.forward()
        return 0.5 * reg_weight * (current_c - self.task_start_c).pow(2)


class HMCLHead(nn.Module):
    """Shared Lorentz linear head supporting fixed or dynamic curvature."""

    def __init__(
        self,
        embed_dim: int,
        curv: float = 0.1,
        dynamic_curvature: bool = False,
        curvature_min: float = 0.05,
        curvature_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.dynamic = dynamic_curvature

        if dynamic_curvature:
            self.curvature_module = DynamicCurvature(
                initial_c=curv,
                c_min=curvature_min,
                c_max=curvature_max,
                learnable=True,
            )
        else:
            self.curvature_module = DynamicCurvature(
                initial_c=curv,
                c_min=curvature_min,
                c_max=curvature_max,
                learnable=False,
            )

        self.head = FullyHyperbolicLinear(embed_dim, curv=curv)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.1).log())

    @property
    def curv(self) -> torch.Tensor:
        return self.curvature_module()

    def set_task_start(self) -> None:
        self.curvature_module.set_task_start()

    def encode_image(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features, curv=self.curv)

    def encode_text(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features, curv=self.curv)

    def log_architecture(self) -> None:
        mode = "dynamic" if self.dynamic else "fixed"
        logger.info(
            "HMCL head: shared fully-hyperbolic linear map ({}), dim={}, curvature={:.4f}",
            mode,
            self.embed_dim,
            float(self.curv.item()),
        )


class EuclideanHead(nn.Module):
    """Euclidean baseline head for verifying the non-Euclidean hypothesis."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.1).log())
        self.is_euclidean = True

    def encode_image(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(features), p=2, dim=-1)

    def encode_text(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(features), p=2, dim=-1)

    def log_architecture(self) -> None:
        logger.info(
            "Euclidean baseline head: shared orthogonal linear map, dim={}",
            self.embed_dim,
        )
