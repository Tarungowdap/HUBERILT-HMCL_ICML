"""Representation Drift and Hierarchy Distortion metrics for Buffer-Free Continual Learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from utils import lorentz as L


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
