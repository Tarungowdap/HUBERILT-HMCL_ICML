"""Lorentz (hyperboloid) geometry used by the hyperbolic methods.

Points are stored by their spatial coordinates x in R^d; the time coordinate is implied by
the hyperboloid constraint x_0 = sqrt(1/c + ||x||^2) for curvature -c.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _as_tensor(c, ref: Tensor) -> Tensor:
    return c if isinstance(c, Tensor) else torch.tensor(c, device=ref.device, dtype=ref.dtype)


def concat_time_dimension(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Lift spatial coordinates [B, d] onto the hyperboloid: returns [B, d+1] = [x_0, x]."""
    c = _as_tensor(curv, x)
    x_time = torch.sqrt(1.0 / c + torch.sum(x ** 2, dim=-1, keepdim=True))
    return torch.cat([x_time, x], dim=-1)


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Lorentzian inner products <x_i, y_j>_L = x_i . y_j - x_i0 * y_j0 for spatial inputs."""
    c = _as_tensor(curv, x)
    x_time = torch.sqrt(1.0 / c + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1.0 / c + torch.sum(y ** 2, dim=-1, keepdim=True))
    return x @ y.T - x_time @ y_time.T


def pairwise_dist(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """Geodesic distances d(x_i, y_j) = arcosh(-c <x_i, y_j>_L) / sqrt(c)."""
    c = _as_tensor(curv, x)
    arg = -c * pairwise_inner(x, y, c)
    return torch.acosh(torch.clamp(arg, min=1.0 + eps)) / torch.sqrt(c)


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
        augmented = concat_time_dimension(features, effective_curv)
        transformed = self.linear(augmented)
        return transformed[..., 1:]


def compute_representation_drift(
    current_feats: torch.Tensor,
    original_feats: torch.Tensor,
    curv: float | torch.Tensor = 0.1,
    is_hyperbolic: bool = True,
) -> dict[str, float]:
    """Measure how much past task representations moved since they were first learned.

    Returns:
        - mean_drift: Average distance between original and current representations.
        - max_drift: Maximum displacement observed.
        - radial_drift: Change in distance from the coordinate origin.
        - angular_drift: Directional drift in normalized angular coordinates.
    """
    if not is_hyperbolic:
        disp = torch.norm(current_feats - original_feats, p=2, dim=-1)
        orig_norm = torch.norm(original_feats, p=2, dim=-1)
        curr_norm = torch.norm(current_feats, p=2, dim=-1)
        radial_disp = torch.abs(curr_norm - orig_norm)

        cos_sim = F.cosine_similarity(current_feats, original_feats, dim=-1)
        angular_disp = 1.0 - cos_sim

        return {
            "mean_drift": float(disp.mean().item()),
            "max_drift": float(disp.max().item()),
            "radial_drift": float(radial_disp.mean().item()),
            "angular_drift": float(angular_disp.mean().item()),
        }

    x_curr = current_feats
    x_orig = original_feats

    curr_time = torch.sqrt(1.0 / curv + torch.sum(x_curr.pow(2), dim=-1, keepdim=True))
    orig_time = torch.sqrt(1.0 / curv + torch.sum(x_orig.pow(2), dim=-1, keepdim=True))

    minkowski_dot = -curr_time * orig_time + torch.sum(x_curr * x_orig, dim=-1, keepdim=True)
    arg = torch.clamp(-curv * minkowski_dot, min=1.0 + 1e-7)
    dist = (1.0 / torch.sqrt(torch.as_tensor(curv))) * torch.log(arg + torch.sqrt(arg.pow(2) - 1.0)).squeeze(-1)

    r_curr = (1.0 / torch.sqrt(torch.as_tensor(curv))) * torch.log(
        torch.clamp(torch.sqrt(torch.as_tensor(curv)) * curr_time, min=1.0 + 1e-7)
    ).squeeze(-1)
    r_orig = (1.0 / torch.sqrt(torch.as_tensor(curv))) * torch.log(
        torch.clamp(torch.sqrt(torch.as_tensor(curv)) * orig_time, min=1.0 + 1e-7)
    ).squeeze(-1)
    radial_disp = torch.abs(r_curr - r_orig)

    cos_sim = F.cosine_similarity(x_curr, x_orig, dim=-1)
    angular_disp = 1.0 - cos_sim

    return {
        "mean_drift": float(dist.mean().item()),
        "max_drift": float(dist.max().item()),
        "radial_drift": float(radial_disp.mean().item()),
        "angular_drift": float(angular_disp.mean().item()),
    }
