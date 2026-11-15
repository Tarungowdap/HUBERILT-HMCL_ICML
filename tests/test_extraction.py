from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import torch

from data_utils.upstream import BACKBONES, default_train_config
from scripts.extract_features import extract_retrieval
from scripts.prepare_data import _referenced_retrieval_images


class _FakeModel:
    device = torch.device("cpu")
    curv = torch.tensor(0.1).log()

    def encode_image(self, images, project):
        assert project
        return torch.ones(images.shape[0], 4)

    def encode_text(self, tokens, project):
        assert project
        return torch.full((len(tokens), 4), 2.0)


def test_official_backbone_specs(tmp_path):
    assert set(BACKBONES) == {"meru_vit_l", "meru_vit_b", "hycoclip_vit_b"}
    assert BACKBONES["meru_vit_l"].package == "meru"
    assert default_train_config("meru_vit_b", tmp_path) == (
        tmp_path / "configs" / "train_meru_vit_b.py"
    )


def test_retrieval_annotation_inventory(tmp_path):
    coco_annotations = tmp_path / "coco" / "annotations"
    coco_annotations.mkdir(parents=True)
    for split in ("train", "val"):
        (coco_annotations / f"captions_{split}2017.json").write_text(
            json.dumps({"images": [{"file_name": f"{split}.jpg"}]})
        )
    flickr = tmp_path / "flickr30k"
    flickr.mkdir()
    (flickr / "dataset_flickr30k.json").write_text(
        json.dumps({"images": [{"filename": "example.jpg"}]})
    )

    assert _referenced_retrieval_images(tmp_path, "coco") == [
        tmp_path / "coco" / "train2017" / "train.jpg",
        tmp_path / "coco" / "val2017" / "val.jpg",
    ]
    assert _referenced_retrieval_images(tmp_path, "flickr30k") == [
        flickr / "flickr30k_images" / "example.jpg"
    ]


def test_retrieval_extraction_writes_features_and_ground_truth(tmp_path):
    instance = {
        "image_id": 7,
        "caption_ids": [70, 71],
        "image": torch.zeros(3, 224, 224),
        "captions": ["caption one", "caption two"],
    }
    api = SimpleNamespace(
        tokenizer=lambda: (lambda texts: [torch.tensor([1])] * len(texts)),
        coco_captions=lambda root, split, transform: [instance],
        flickr30k_captions=lambda root, split, transform: [instance],
    )

    extract_retrieval(
        api=api,
        model=_FakeModel(),
        dataset="coco",
        split="test",
        data_root=tmp_path / "data",
        output_root=tmp_path / "features",
        backbone="meru_vit_b",
        checkpoint_sha256="abc123",
        image_size=224,
    )

    image_path = tmp_path / "features" / "coco" / "coco_test_image.h5"
    text_path = tmp_path / "features" / "coco" / "coco_test_text.h5"
    with h5py.File(image_path) as handle:
        assert handle["features"].shape == (1, 4)
        assert handle["ids"][:].tolist() == [7]
        assert handle.attrs["checkpoint_sha256"] == "abc123"
    with h5py.File(text_path) as handle:
        assert handle["features"].shape == (2, 4)
        assert handle["ids"][:].tolist() == [70, 71]

    ground_truth = json.loads(
        (tmp_path / "features" / "coco" / "coco_test_retrieval_gt.json").read_text()
    )
    assert ground_truth["image_to_text_gt"] == {"7": [70, 71]}
    assert ground_truth["text_to_image_gt"] == {"70": [7], "71": [7]}
