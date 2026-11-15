import torch

from main import REPORTED_RETRIEVAL_K, _resolve_pullback_beta, _resolve_variance_ratio
from trainer.head import HMCLHead
from trainer.trainer import ContinualTrainer
from utils.config import ClassificationConfig, RetrievalConfig


def _toy_task(offset: float):
    generator = torch.Generator().manual_seed(int(offset * 10) + 7)
    image = torch.randn(8, 4, generator=generator) + offset
    text = image + 0.01 * torch.randn(8, 4, generator=generator)
    return {
        "features": image,
        "text_features": text,
        "labels": torch.arange(8) % 2,
    }


def test_shared_head_is_one_module():
    model = HMCLHead(embed_dim=4, curv=0.1)
    features = torch.randn(2, 4)
    assert torch.equal(model.encode_image(features), model.encode_text(features))
    assert torch.allclose(model.logit_scale.exp(), torch.tensor(10.0))


def test_two_task_cpu_smoke():
    classification = ClassificationConfig(
        device="cpu", epochs=1, batch_size=4, entail_weight=0.0
    )
    retrieval = RetrievalConfig(
        device="cpu", epochs=1, batch_size=4, entail_weight=0.0
    )
    trainer = ContinualTrainer(classification, retrieval, variance_ratio=0.8)

    model = trainer.fit_task(_toy_task(0.0), task_type="classification")
    model = trainer.fit_task(_toy_task(1.0), model=model, task_type="classification")

    assert trainer.task_id == 2
    assert trainer.projector is not None
    assert trainer.projector.n_samples == 32
    assert torch.isfinite(model.head.linear.weight).all()


def test_pullback_interpolates_only_head_matrices():
    model = HMCLHead(embed_dim=4, curv=0.1)
    anchor = ContinualTrainer._snapshot_head_matrix(model)
    before_scale = model.logit_scale.detach().clone()
    anchor_weight = model.head.linear.weight.detach().clone()
    with torch.no_grad():
        model.head.linear.weight.add_(2.0)

    ContinualTrainer._apply_head_matrix_pullback(model, anchor, beta=0.25)

    assert torch.allclose(model.head.linear.weight, anchor_weight + 1.5)
    assert torch.allclose(model.logit_scale, before_scale)


def test_retrieval_training_selects_one_matching_caption_per_image():
    text_features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    train = {
        "image_ids": [10, 20],
        "text_ids": [100, 101, 200, 201],
        "image_to_text_gt": {10: {100, 101}, 20: {200, 201}},
    }

    aligned = ContinualTrainer._align_retrieval_text(train, text_features)

    assert aligned.tolist() == [[1.0], [3.0]]


def test_retrieval_memory_keeps_all_captions():
    classification = ClassificationConfig(
        device="cpu", epochs=1, batch_size=2, entail_weight=0.0
    )
    retrieval = RetrievalConfig(
        device="cpu", epochs=1, batch_size=2, entail_weight=0.0
    )
    trainer = ContinualTrainer(classification, retrieval, variance_ratio=0.8)
    train = {
        "features": torch.randn(2, 4),
        "text_features": torch.randn(4, 4),
        "image_ids": [10, 20],
        "text_ids": [100, 101, 200, 201],
        "image_to_text_gt": {10: {100, 101}, 20: {200, 201}},
    }

    trainer.fit_task(train, task_type="retrieval")

    assert trainer.projector is not None
    assert trainer.projector.n_samples == 6


def test_reported_pullback_and_retrieval_defaults():
    assert _resolve_pullback_beta("meru_vit_l", None) == 0.10
    assert _resolve_pullback_beta("meru_vit_b", None) == 0.10
    assert _resolve_pullback_beta("hycoclip_vit_b", None) == 0.20
    assert _resolve_variance_ratio("meru_vit_l", None) == 0.9
    assert _resolve_variance_ratio("meru_vit_b", None) == 0.8
    assert _resolve_variance_ratio("hycoclip_vit_b", None) == 0.9
    assert REPORTED_RETRIEVAL_K == {
        "meru_vit_l": 5,
        "meru_vit_b": 5,
        "hycoclip_vit_b": 1,
    }
