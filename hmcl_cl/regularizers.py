"""Forgetting regularisers: EWC (Kirkpatrick et al., 2017) and HMCL-style gradient projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class EWC:
    """Computes and accumulates diagonal Fisher information matrix for continual learning."""

    def __init__(self, ewc_lambda: float = 100.0, ewc_gamma: float = 1.0) -> None:
        self.ewc_lambda = float(ewc_lambda)
        self.ewc_gamma = float(ewc_gamma)
        self.fisher_diag: dict[str, torch.Tensor] = {}
        self.star_params: dict[str, torch.Tensor] = {}

    def is_active(self) -> bool:
        """Returns True if at least one historical task checkpoint has been recorded."""
        return len(self.star_params) > 0

    def compute_fisher(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion_fn,
        device: torch.device,
        max_samples: int = 2048,
    ) -> None:
        """Calculate diagonal empirical Fisher info matrix at the end of a task."""
        model.eval()
        fisher_accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device=device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        sample_count = 0
        for batch in train_loader:
            model.zero_grad()
            loss = criterion_fn(model, batch)
            loss.backward()

            batch_size = (
                batch["image_features"].size(0)
                if isinstance(batch, dict) and "image_features" in batch
                else 1
            )
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_accum[name].add_(p.grad.detach().pow(2) * batch_size)

            sample_count += batch_size
            if sample_count >= max_samples:
                break

        if sample_count > 0:
            for name in fisher_accum:
                fisher_accum[name].div_(sample_count)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.fisher_diag:
                self.fisher_diag[name] = (
                    self.ewc_gamma * self.fisher_diag[name] + fisher_accum[name]
                )
            else:
                self.fisher_diag[name] = fisher_accum[name]

            self.star_params[name] = p.detach().clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss: 0.5 * lambda * sum(F_i * (theta_i - theta*_i)^2)."""
        if not self.is_active():
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self.fisher_diag and name in self.star_params:
                fisher = self.fisher_diag[name].to(p.device)
                star = self.star_params[name].to(p.device)
                loss = loss + torch.sum(fisher * (p - star).pow(2))

        return 0.5 * self.ewc_lambda * loss


class MemoryProjector:
    """Maintain one image-text covariance and project updates off its PCA span."""

    def __init__(self, embed_dim: int, device: torch.device) -> None:
        self.embed_dim = int(embed_dim)
        self.device = device
        self.effective_covariance = torch.zeros(
            embed_dim, embed_dim, device=device
        )
        self.n_samples = 0

    @torch.no_grad()
    def update_covariance(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> None:
        """Accumulate the uncentered covariance of both modalities' space parts."""
        combined = torch.cat(
            [image_features.to(self.device), text_features.to(self.device)], dim=0
        )
        if combined.ndim != 2 or combined.shape[1] != self.embed_dim + 1:
            raise ValueError(
                f"Expected augmented features [N, {self.embed_dim + 1}], "
                f"got {tuple(combined.shape)}"
            )
        space = combined[:, 1:]
        covariance_sum = space.T @ space
        self.effective_covariance.add_(
            0.5 * (covariance_sum + covariance_sum.T)
        )
        self.n_samples += combined.shape[0]

    @torch.no_grad()
    def get_covariance(self) -> torch.Tensor:
        if self.n_samples == 0:
            return torch.zeros_like(self.effective_covariance)
        covariance = self.effective_covariance / float(self.n_samples)
        identity = torch.eye(
            self.embed_dim, device=covariance.device, dtype=covariance.dtype
        )
        return 0.5 * (covariance + covariance.T) + 1e-6 * identity

    @torch.no_grad()
    def calculate_projector(
        self, covariance: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Return the PCA span explaining at least ``epsilon`` total variance."""
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]
        total_variance = eigenvalues.sum()
        if total_variance < 1e-10:
            return torch.zeros_like(covariance)

        cumulative = eigenvalues.cumsum(dim=0) / total_variance
        components = int((cumulative < epsilon).sum().item()) + 1
        components = min(components, covariance.shape[0])
        basis = eigenvectors[:, :components]
        return basis @ basis.T

    def project_gradient(
        self, gradient: torch.Tensor, projector: torch.Tensor
    ) -> torch.Tensor:
        """Remove preserved spatial directions and freeze the Lorentz time column."""
        if gradient.ndim != 2 or gradient.shape[1] != self.embed_dim + 1:
            raise ValueError(
                f"Expected Lorentz weight gradient [D+1, {self.embed_dim + 1}], "
                f"got {tuple(gradient.shape)}"
            )
        space_gradient = gradient[:, 1:]
        projected_space = space_gradient - space_gradient @ projector
        return torch.cat([torch.zeros_like(gradient[:, :1]), projected_space], dim=1)
