"""Shared feature covariance and gradient projection for HMCL."""

import torch
from loguru import logger


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
        logger.info(
            "HMCL memory: target={:.2%}, actual={:.2%}, rank={}/{}",
            epsilon,
            float(cumulative[components - 1]),
            components,
            covariance.shape[0],
        )
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
