"""Unit tests for ContinualTrainer end-to-end task progression."""

import pytest
import torch
from trainer.trainer import ContinualTrainer
from utils.config import ClassificationConfig, RetrievalConfig


def test_trainer_classification_and_retrieval_steps():
    cls_cfg = ClassificationConfig(epochs=1, batch_size=16, lr=1e-3, device="cpu")
    ret_cfg = RetrievalConfig(epochs=1, batch_size=16, lr=1e-3, device="cpu")

    trainer = ContinualTrainer(
        cls_cfg,
        ret_cfg,
        method="dynamic-curvature",
        curvature_min=0.05,
        curvature_max=0.5,
        curvature_lr=1e-3,
    )

    dim = 16
    train_cls = {
        "image_features": torch.randn(32, dim),
        "text_features": torch.randn(32, dim),
        "labels": torch.randint(0, 4, (32,)),
    }
    test_cls = {
        "image_features": torch.randn(16, dim),
        "labels": torch.randint(0, 4, (16,)),
    }

    model = trainer.fit_task(train_cls, None, "classification")
    assert model is not None

    class_text = trainer.build_class_text(train_cls["text_features"], train_cls["labels"])
    acc = trainer.evaluate_classification(model, test_cls, class_text)
    assert 0.0 <= acc <= 100.0

    train_ret = {
        "image_features": torch.randn(32, dim),
        "text_features": torch.randn(32, dim),
        "gt_pairs": [(i, i) for i in range(32)],
    }
    test_ret = {
        "image_features": torch.randn(16, dim),
        "text_features": torch.randn(16, dim),
        "gt_pairs": [(i, i) for i in range(16)],
    }

    model = trainer.fit_task(train_ret, model, "retrieval")
    ret_metrics = trainer.evaluate_retrieval(model, test_ret)
    assert "i2t_r1" in ret_metrics
    assert "t2i_r5" in ret_metrics
    assert 0.0 <= ret_metrics["i2t_r5"] <= 100.0
