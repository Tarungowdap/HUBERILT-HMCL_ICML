"""Adapted from MERU (https://github.com/facebookresearch/meru).
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def arcosh(x: Tensor, eps: float = 1e-7) -> Tensor:
    """Numerically stable arcosh for clamped input >= 1.0 + eps."""
    clamped = torch.clamp(x, min=1.0 + eps)
    return torch.acosh(clamped)


def minkowski_dot(u: Tensor, v: Tensor) -> Tensor:
    """Minkowski / Lorentzian inner product <u, v>_L = -u_0 * v_0 + <u_space, v_space>."""
    if u.dim() == 1 and v.dim() == 1:
        return -u[0] * v[0] + torch.dot(u[1:], v[1:])
    time_part = -u[..., 0] * v[..., 0]
    space_part = torch.sum(u[..., 1:] * v[..., 1:], dim=-1)
    return time_part + space_part


def project_to_lorentz(x: Tensor, c: float | Tensor = 1.0) -> Tensor:
    """Project tensor with space coordinates or arbitrary (D+1) coordinates onto H^d(c)."""
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    if x.shape[-1] > 1:
        space = x[..., 1:] if x.shape[-1] > 1 and x[..., 0:1].mean() > 0 else x
    else:
        space = x
    return concat_time_dimension(space, curv_val)


def exp_map_origin(v: Tensor, c: float | Tensor = 1.0) -> Tensor:
    """Riemannian exponential map at origin of Lorentz hyperboloid H^d(c).

    Maps tangent vector v in R^d (tangent space T_0 H) to point x in H^d(c) subset R^{d+1}.
    """
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    sqrt_c = curv_val ** 0.5
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)

    scaled_norm = sqrt_c * v_norm
    time = torch.cosh(scaled_norm) / sqrt_c
    space = torch.sinh(scaled_norm) / (sqrt_c * v_norm) * v

    zero_mask = (v_norm < 1e-7)
    if zero_mask.any():
        space = torch.where(zero_mask, v, space)
        time = torch.where(zero_mask, torch.full_like(time, 1.0 / sqrt_c), time)

    return torch.cat([time, space], dim=-1)


def log_map_origin(x: Tensor, c: float | Tensor = 1.0) -> Tensor:
    """Riemannian logarithmic map at origin of Lorentz hyperboloid H^d(c).

    Maps point x in H^d(c) (shape [..., d+1]) to tangent vector v in R^d.
    """
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    sqrt_c = curv_val ** 0.5
    time = x[..., 0:1]
    space = x[..., 1:]

    cosh_arg = (sqrt_c * time).clamp(min=1.0 + 1e-7)
    dist = torch.acosh(cosh_arg)

    sinh_val = torch.sinh(dist).clamp(min=1e-8)
    scale = dist / sinh_val

    zero_mask = (dist < 1e-6)
    v = space * scale
    if zero_mask.any():
        v = torch.where(zero_mask, space, v)

    return v


def lorentz_distance(u: Tensor, v: Tensor, c: float | Tensor = 1.0, eps: float = 1e-7) -> Tensor:
    """Geodesic distance between points on Lorentz hyperboloid H^d(c)."""
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    inner = minkowski_dot(u, v)
    c_inner = (-curv_val * inner).clamp(min=1.0 + eps)
    return torch.acosh(c_inner) / (curv_val ** 0.5)


def pairwise_lorentz_distance(x: Tensor, y: Tensor, c: float | Tensor = 1.0, eps: float = 1e-7) -> Tensor:
    """Pairwise geodesic distance between two batches of points in H^d(c) (shape [B, D+1])."""
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    time_prod = x[:, 0:1] @ y[:, 0:1].T
    space_prod = x[:, 1:] @ y[:, 1:].T
    inner = space_prod - time_prod
    c_inner = (-curv_val * inner).clamp(min=1.0 + eps)
    return torch.acosh(c_inner) / (curv_val ** 0.5)


def concat_time_dimension(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Concatenate the time dimension to the input tensor x of shape (B, D)."""
    curv_t = curv if isinstance(curv, Tensor) else torch.tensor(curv, device=x.device, dtype=x.dtype)
    x_time = torch.sqrt(1.0 / curv_t + torch.sum(x**2, dim=-1, keepdim=True))
    return torch.cat([x_time, x], dim=-1)


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Compute pairwise Lorentzian inner product between space-only input vectors."""
    curv_t = curv if isinstance(curv, Tensor) else torch.tensor(curv, device=x.device, dtype=x.dtype)
    x_time = torch.sqrt(1.0 / curv_t + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1.0 / curv_t + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """Compute pairwise distance along geodesics connecting space-only inputs."""
    curv_t = curv if isinstance(curv, Tensor) else torch.tensor(curv, device=x.device, dtype=x.dtype)
    c_xyl = -curv_t * pairwise_inner(x, y, curv_t)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1.0 + eps))
    return _distance / torch.sqrt(curv_t)


def half_aperture(x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8) -> Tensor:
    """Half aperture angle of entailment cone formed by vectors on hyperboloid."""
    curv_val = float(curv.item()) if isinstance(curv, Tensor) else float(curv)
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * (curv_val**0.5) + eps)
    return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """Exterior angle at x in hyperbolic triangle Oxy."""
    curv_val = float(curv.item()) if isinstance(curv, Tensor) else float(curv)
    x_time = torch.sqrt(1 / curv_val + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv_val + torch.sum(y**2, dim=-1))
    c_xyl = curv_val * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


def compute_entailment_loss(text_feats: Tensor, image_feats: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Compute hyperbolic entailment loss: text should entail matching image."""
    curv_val = float(curv.item()) if isinstance(curv, Tensor) else float(curv)
    _angle = oxy_angle(text_feats, image_feats, curv_val)
    _aperture = half_aperture(text_feats, curv_val)
    return torch.clamp(_angle - _aperture, min=0).mean()


def hyperbolic_entailment_loss(u: Tensor, v: Tensor, c: float | Tensor = 1.0) -> Tensor:
    """Alias for entailment cone loss with [B, D+1] or [B, D] inputs."""
    curv_val = float(c.item()) if isinstance(c, Tensor) else float(c)
    u_space = u[..., 1:] if u.shape[-1] > v.shape[-1] or u.dim() > 1 and u.shape[-1] > 8 else u
    v_space = v[..., 1:] if v.shape[-1] > u.shape[-1] or v.dim() > 1 and v.shape[-1] > 8 else v
    if u_space.shape[-1] == u.shape[-1] and u.shape[-1] != 8:
        pass
    else:
        u_space = u[..., 1:]
        v_space = v[..., 1:]
    return compute_entailment_loss(u_space, v_space, curv_val)
