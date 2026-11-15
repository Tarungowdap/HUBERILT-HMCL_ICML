"""Poincaré Ball Model of Hyperbolic Geometry.

Provides diffeomorphic stereographic projections to/from the Lorentz hyperboloid,
Poincaré geodesic distances, Möbius addition, and conformal factors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lorentz_to_poincare(x_lorentz: torch.Tensor, curv: float | torch.Tensor = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """Project points from the Lorentz hyperboloid to the Poincaré ball via stereographic projection.

    x_lorentz: [..., D+1] where x_lorentz[..., 0] is the time component.
    Returns: [..., D] in Poincaré ball B^d(c) = {p in R^d : c * ||p||^2 < 1}.
    """
    if not isinstance(curv, torch.Tensor):
        curv = torch.tensor(curv, dtype=x_lorentz.dtype, device=x_lorentz.device)
    sqrt_c = torch.sqrt(torch.clamp(curv, min=eps))

    time_comp = x_lorentz[..., :1]
    space_comp = x_lorentz[..., 1:]

    denom = torch.clamp(sqrt_c * time_comp + 1.0, min=eps)
    return space_comp / denom


def poincare_to_lorentz(p: torch.Tensor, curv: float | torch.Tensor = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """Project points from Poincaré ball to the Lorentz hyperboloid."""
    if not isinstance(curv, torch.Tensor):
        curv = torch.tensor(curv, dtype=p.dtype, device=p.device)
    sqrt_c = torch.sqrt(torch.clamp(curv, min=eps))

    norm_sq = torch.sum(p.pow(2), dim=-1, keepdim=True)
    norm_sq = torch.clamp(norm_sq, max=(1.0 - eps) / torch.clamp(curv, min=eps))

    denom = torch.clamp(1.0 - curv * norm_sq, min=eps)
    time_comp = (1.0 / sqrt_c) * (1.0 + curv * norm_sq) / denom
    space_comp = 2.0 * p / denom

    return torch.cat([time_comp, space_comp], dim=-1)


def poincare_dist(u: torch.Tensor, v: torch.Tensor, curv: float | torch.Tensor = 1.0, eps: float = 1e-7) -> torch.Tensor:
    """Poincaré geodesic distance between u and v."""
    if not isinstance(curv, torch.Tensor):
        curv = torch.tensor(curv, dtype=u.dtype, device=u.device)
    sqrt_c = torch.sqrt(torch.clamp(curv, min=eps))

    diff_sq = torch.sum((u - v).pow(2), dim=-1)
    u_sq = torch.sum(u.pow(2), dim=-1).clamp(max=(1.0 - eps) / torch.clamp(curv, min=eps))
    v_sq = torch.sum(v.pow(2), dim=-1).clamp(max=(1.0 - eps) / torch.clamp(curv, min=eps))

    denom = torch.clamp((1.0 - curv * u_sq) * (1.0 - curv * v_sq), min=eps)
    alpha = 1.0 + 2.0 * curv * diff_sq / denom
    alpha_clamped = torch.clamp(alpha, min=1.0 + eps)

    return (1.0 / sqrt_c) * torch.log(alpha_clamped + torch.sqrt(alpha_clamped.pow(2) - 1.0))
