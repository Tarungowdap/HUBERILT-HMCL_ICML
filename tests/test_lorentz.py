"""Unit tests for Lorentz hyperbolic geometry operations."""

import pytest
import torch
from utils.lorentz import (
    arcosh,
    exp_map_origin,
    hyperbolic_entailment_loss,
    log_map_origin,
    lorentz_distance,
    minkowski_dot,
    pairwise_lorentz_distance,
)


def test_minkowski_dot():
    u = torch.tensor([2.0, 1.0, 1.0])
    v = torch.tensor([3.0, 2.0, 1.0])
    dot = minkowski_dot(u, v)
    assert torch.isclose(dot, torch.tensor(-3.0))


def test_exp_and_log_map_origin():
    c = 0.1
    dim = 16
    v = torch.randn(10, dim) * 0.5
    x = exp_map_origin(v, c=c)
    assert x.shape == (10, dim + 1)

    inner = minkowski_dot(x, x)
    expected = torch.full_like(inner, -1.0 / c)
    assert torch.allclose(inner, expected, atol=1e-4)

    v_rec = log_map_origin(x, c=c)
    assert torch.allclose(v, v_rec, atol=1e-4)


def test_lorentz_distance_properties():
    c = 0.2
    dim = 8
    v1 = torch.randn(5, dim) * 0.3
    v2 = torch.randn(5, dim) * 0.3

    x1 = exp_map_origin(v1, c=c)
    x2 = exp_map_origin(v2, c=c)

    d_self = lorentz_distance(x1, x1, c=c)
    assert torch.all(d_self >= 0.0)
    assert torch.allclose(d_self, torch.zeros_like(d_self), atol=2e-3)

    d12 = lorentz_distance(x1, x2, c=c)
    d21 = lorentz_distance(x2, x1, c=c)
    assert torch.allclose(d12, d21, atol=1e-5)


def test_pairwise_lorentz_distance():
    c = 0.1
    dim = 10
    v1 = torch.randn(4, dim) * 0.2
    v2 = torch.randn(6, dim) * 0.2

    x1 = exp_map_origin(v1, c=c)
    x2 = exp_map_origin(v2, c=c)

    pwd = pairwise_lorentz_distance(x1, x2, c=c)
    assert pwd.shape == (4, 6)

    for i in range(4):
        for j in range(6):
            d_single = lorentz_distance(x1[i], x2[j], c=c)
            assert torch.isclose(pwd[i, j], d_single, atol=1e-4)


def test_hyperbolic_entailment_loss():
    c = 0.1
    u = exp_map_origin(torch.zeros(4, 8), c=c)
    v = exp_map_origin(torch.randn(4, 8) * 0.5, c=c)

    loss = hyperbolic_entailment_loss(u, v, c=c)
    assert loss.item() >= 0.0
