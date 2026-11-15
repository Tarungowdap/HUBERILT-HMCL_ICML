"""Unit tests for Dynamic Curvature and Task Anchoring."""

import pytest
import torch
from models.lorentz_head import DynamicCurvature, SharedLorentzHead


def test_dynamic_curvature_bounds():
    c_min = 0.05
    c_max = 0.5
    curv = DynamicCurvature(initial_c=0.1, c_min=c_min, c_max=c_max, learnable=True)

    c_val = curv()
    assert c_min <= c_val.item() <= c_max
    assert abs(c_val.item() - 0.1) < 1e-3

    with torch.no_grad():
        curv.raw_c.copy_(torch.tensor(100.0))
    assert curv().item() <= c_max

    with torch.no_grad():
        curv.raw_c.copy_(torch.tensor(-100.0))
    assert curv().item() >= c_min


def test_dynamic_curvature_gradient_and_anchor():
    curv = DynamicCurvature(initial_c=0.1, c_min=0.05, c_max=0.5, learnable=True)
    curv.set_task_start()

    anchor_loss = curv.get_anchor_loss(reg_weight=1.0)
    assert anchor_loss.item() == 0.0

    with torch.no_grad():
        curv.raw_c.add_(0.5)

    anchor_loss = curv.get_anchor_loss(reg_weight=2.0)
    assert anchor_loss.item() > 0.0

    anchor_loss.backward()
    assert curv.raw_c.grad is not None
    assert torch.abs(curv.raw_c.grad) > 0.0


def test_shared_lorentz_head_pullback():
    dim = 16
    head = SharedLorentzHead(feature_dim=dim, initial_curv=0.1)
    head.set_task_start()

    with torch.no_grad():
        head.weight.add_(torch.ones_like(head.weight))

    start_weight = head.task_start_weight.clone()
    w_before = head.weight.clone()
    head.apply_pullback(beta=0.2)
    expected = 0.8 * w_before + 0.2 * start_weight
    assert torch.allclose(head.weight, expected, atol=1e-5)
