"""Unit tests for EWC baseline."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainer.ewc import EWC


def test_ewc_accumulation_and_penalty():
    ewc = EWC(ewc_lambda=50.0, ewc_gamma=0.9)
    assert not ewc.is_active()

    model = nn.Linear(8, 4)
    x = torch.randn(20, 8)
    y = torch.randn(20, 4)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)

    def criterion(m, batch):
        bx, by = batch
        return torch.nn.functional.mse_loss(m(bx), by)

    ewc.compute_fisher(model, loader, criterion, device=torch.device("cpu"), max_samples=20)
    assert ewc.is_active()
    assert "weight" in ewc.fisher_diag
    assert "bias" in ewc.fisher_diag

    pen_zero = ewc.penalty(model)
    assert torch.isclose(pen_zero, torch.tensor(0.0), atol=1e-5)

    with torch.no_grad():
        model.weight.add_(0.1)
    pen_positive = ewc.penalty(model)
    assert pen_positive.item() > 0.0
