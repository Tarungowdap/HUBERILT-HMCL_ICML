"""Shared Lorentz Head with learnable constrained curvature and task anchoring."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from utils.lorentz import exp_map_origin, log_map_origin, project_to_lorentz


class DynamicCurvature(nn.Module):
    """Constrained curvature parameterization for hyperbolic spaces.

    Supports:
    - Fixed curvature (HMCL baseline)
    - Learnable curvature bounded in [c_min, c_max] via sigmoid parameterization
    - Soft task-start anchoring loss: 0.5 * reg * (c - c_start)^2
    - Task-wise curvature trajectory logging
    """

    def __init__(
        self,
        initial_c: float = 0.1,
        c_min: float = 0.05,
        c_max: float = 0.5,
        learnable: bool = False,
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
        """Return the effective curvature c in [c_min, c_max]."""
        if self.learnable:
            sig = torch.sigmoid(self.raw_c)
            return self.c_min + (self.c_max - self.c_min) * sig
        else:
            sig = torch.sigmoid(self.raw_c)
            return self.c_min + (self.c_max - self.c_min) * sig

    def set_task_start(self) -> None:
        """Anchor the current curvature as the task reference."""
        with torch.no_grad():
            self.task_start_c.copy_(self.forward())

    def get_anchor_loss(self, reg_weight: float) -> torch.Tensor:
        """Calculate soft anchor loss to restrict curvature drift: 0.5 * reg * (c - c_start)^2."""
        if not self.learnable or reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.raw_c.device)
        current_c = self.forward()
        return 0.5 * reg_weight * (current_c - self.task_start_c).pow(2)


class SharedLorentzHead(nn.Module):
    """Shared Lorentz linear adaptation head with covariance projection support.

    Transforms frozen multimodal representations via linear weight W,
    maps them onto the Lorentz hyperboloid H^d(c), and maintains task checkpoints
    for HMCL pullback updates:
        W <- (1 - beta) * W + beta * W_task_start.
    """

    def __init__(
        self,
        feature_dim: int,
        initial_curv: float = 0.1,
        curvature_min: float = 0.05,
        curvature_max: float = 0.5,
        learnable_curvature: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        self.weight = nn.Parameter(torch.eye(feature_dim, dtype=torch.float32))

        self.curvature = DynamicCurvature(
            initial_c=initial_curv,
            c_min=curvature_min,
            c_max=curvature_max,
            learnable=learnable_curvature,
        )

        self.register_buffer("task_start_weight", torch.eye(feature_dim, dtype=torch.float32))

    @property
    def current_curvature(self) -> torch.Tensor:
        return self.curvature()

    def set_task_start(self) -> None:
        """Store weight and curvature snapshots at task start."""
        with torch.no_grad():
            self.task_start_weight.copy_(self.weight.data)
            self.curvature.set_task_start()

    def apply_pullback(self, beta: float) -> None:
        """Apply HMCL pullback update: W <- (1 - beta) * W + beta * W_task_start."""
        if beta <= 0.0:
            return
        with torch.no_grad():
            self.weight.data.mul_(1.0 - beta).add_(self.task_start_weight, alpha=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward input through shared weight matrix and project onto Lorentz hyperboloid.

        x: [B, D] tangent or Euclidean features (or [B, D+1] Lorentz features).
        Returns: [B, D+1] points in Lorentz space H^d(c).
        """
        c = self.current_curvature
        if x.shape[-1] == self.feature_dim + 1:
            x_tangent = log_map_origin(x, c)
        else:
            x_tangent = x

        adapted = torch.matmul(x_tangent, self.weight.t())

        lorentz_points = exp_map_origin(adapted, c)
        return lorentz_points
