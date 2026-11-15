"""Elastic Weight Consolidation (EWC) with diagonal empirical Fisher information."""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


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

        logger.info(
            "Recorded EWC Fisher checkpoint over {} samples (lambda={:.1f}, gamma={:.2f})",
            sample_count,
            self.ewc_lambda,
            self.ewc_gamma,
        )

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
