"""Covariance-Memory Gradient Projector for HMCL.

Constructs an orthonormal subspace basis from task representation covariances
using a variance ratio threshold (PCA), and projects weight gradients onto the
orthogonal complement to prevent interference with historical tasks:
    G_proj = G - G * U * U^T.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class CovarianceMemoryProjector:
    """Maintains covariance subspace memory and projects gradients."""

    def __init__(self, variance_ratio: float = 0.9, device: torch.device | str = "cpu") -> None:
        self.variance_ratio = float(variance_ratio)
        self.device = torch.device(device)
        self.basis: torch.Tensor | None = None

    @property
    def memory_size(self) -> int:
        return self.basis.shape[1] if self.basis is not None else 0

    def update_memory(self, features: torch.Tensor) -> None:
        """Update subspace memory using PCA on the covariance of the task's representations.

        features: [N, D] representations from the completed task.
        """
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = features.detach().to(self.device, dtype=torch.float32)

        n_samples, dim = features.shape
        mean = torch.mean(features, dim=0, keepdim=True)
        centered = features - mean
        cov = torch.mm(centered.t(), centered) / max(1, n_samples - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = torch.sum(eigenvalues).clamp(min=1e-8)
        cum_var = torch.cumsum(eigenvalues, dim=0) / total_var

        k = int(torch.searchsorted(cum_var, self.variance_ratio).item()) + 1
        k = max(1, min(k, dim))
        new_basis = eigenvectors[:, :k]

        if self.basis is None:
            self.basis = new_basis
            logger.info(
                "Initialized covariance memory with {} basis vectors (variance ratio: {:.2f})",
                self.basis.shape[1],
                self.variance_ratio,
            )
        else:
            proj = torch.mm(self.basis, torch.mm(self.basis.t(), new_basis))
            orth_new = new_basis - proj

            q, r = torch.linalg.qr(orth_new)
            diag_r = torch.abs(torch.diag(r))
            valid = diag_r > 1e-4
            if torch.any(valid):
                q_valid = q[:, valid]
                self.basis = torch.cat([self.basis, q_valid], dim=1)
                self.basis, _ = torch.linalg.qr(self.basis)
                logger.info(
                    "Updated covariance memory: added {} vectors, total basis dimension: {}/{}",
                    q_valid.shape[1],
                    self.basis.shape[1],
                    dim,
                )
            else:
                logger.info(
                    "New task subspace fully spanned by existing memory ({}/{} basis vectors)",
                    self.basis.shape[1],
                    dim,
                )

    def project_gradient(self, weight_param: nn.Parameter) -> None:
        """Project weight gradient G onto orthogonal complement of the memory subspace:

            G_proj = G - G * U * U^T.

        Leaves curvature and other non-matrix parameters untouched.
        """
        if self.basis is None or weight_param.grad is None:
            return

        with torch.no_grad():
            grad = weight_param.grad.data
            basis = self.basis.to(grad.device)
            proj = torch.mm(torch.mm(grad, basis), basis.t())
            weight_param.grad.data.sub_(proj)
