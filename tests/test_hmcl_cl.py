"""Unit tests for the hmcl_cl package (fast, CPU, no downloads)."""

import torch

from hmcl_cl import (EWC, ContinualLearningModel, DynamicCurvature, GPUAugment, MemoryProjector,
                     METHOD_ALIASES, make_args, pairwise_dist, run_experiment)
from hmcl_cl.geometry import concat_time_dimension, pairwise_inner


def test_points_lie_on_hyperboloid():
    x, c = torch.randn(5, 4), 0.3
    p = concat_time_dimension(x, c)
    inner = -p[:, 0] ** 2 + (p[:, 1:] ** 2).sum(-1)
    assert torch.allclose(inner, torch.full((5,), -1 / c), atol=1e-4)


def test_distance_is_a_metric():
    x, y = torch.randn(6, 3), torch.randn(4, 3)
    d = pairwise_dist(x, y, 0.5)
    assert d.shape == (6, 4) and (d >= 0).all()
    assert torch.allclose(pairwise_dist(x, x, 0.5).diagonal(), torch.zeros(6), atol=1e-3)
    assert torch.allclose(d, pairwise_dist(y, x, 0.5).T, atol=1e-5)
    assert torch.allclose(pairwise_inner(x, y, 0.5), pairwise_inner(y, x, 0.5).T, atol=1e-5)


def test_dynamic_curvature_bounds_and_learnability():
    dc = DynamicCurvature(0.1, 0.05, 1.0, learnable=True)
    assert 0.05 <= float(dc()) <= 1.0 and dc.raw_c.requires_grad
    fixed = DynamicCurvature(0.1, 0.05, 1.0, learnable=False)
    assert not any(p.requires_grad for p in fixed.parameters())
    assert abs(float(fixed()) - 0.1) < 1e-4


def test_ewc_penalty_zero_at_anchor_positive_after_move():
    model = torch.nn.Linear(4, 2)
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    ewc = EWC(ewc_lambda=10.0)
    loader = [(torch.arange(i, i + 4),) for i in range(0, 16, 4)]
    ewc.compute_fisher(model, loader, lambda m, b: torch.nn.functional.cross_entropy(m(x[b[0]]), y[b[0]]),
                       torch.device("cpu"))
    assert float(ewc.penalty(model)) == 0.0
    with torch.no_grad():
        model.weight.add_(0.5)
    assert float(ewc.penalty(model)) > 0.0


def test_projector_removes_protected_directions():
    proj = MemoryProjector(3, torch.device("cpu"))
    feats = torch.zeros(50, 4); feats[:, 1] = torch.randn(50)
    proj.update_covariance(feats, feats)
    P = proj.calculate_projector(proj.get_covariance(), 0.9)
    g = proj.project_gradient(torch.ones(4, 4), P)
    assert torch.allclose(g[:, 0], torch.zeros(4)) and torch.allclose(g[:, 1], torch.zeros(4), atol=1e-5)


def test_all_methods_build_and_encode():
    x = torch.randn(2, 3, 32, 32)
    for m in set(METHOD_ALIASES.values()):
        z = ContinualLearningModel(m, 16, 8).encode(x)
        assert z.shape == (2, 16) and torch.isfinite(z).all()


def test_tiny_end_to_end_run():
    g = torch.Generator().manual_seed(0)
    X = (torch.rand(48, 3, 32, 32, generator=g) * 255).to(torch.uint8)
    Y = torch.arange(48) % 4
    stream = {f"task_{t+1}": dict(train_x=X[(Y // 2) == t], train_y=Y[(Y // 2) == t] - 2 * t,
                                   test_x=X[(Y // 2) == t], test_y=Y[(Y // 2) == t] - 2 * t,
                                   num_classes=2, class_offset=2 * t) for t in range(2)}
    aug = GPUAugment((0.5,) * 3, (0.25,) * 3)
    r = run_experiment("dynamic-curvature", stream, make_args(epochs=1, embed_dim=8, width=8, batch_size=8),
                       0, torch.device("cpu"), aug)
    for p in ("task_il", "class_il"):
        assert 0 <= r[p]["final_accuracy"] <= 100 and len(r[p]["score_matrix"]) == 2
    assert len(r["curvature_history"]) == 2
