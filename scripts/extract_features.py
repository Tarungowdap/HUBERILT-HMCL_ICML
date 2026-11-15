#!/usr/bin/env python3
"""Extract frozen MERU/HyCoCLIP representations into HMCL's HDF5 layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.upstream import (
    BACKBONES,
    default_train_config,
    load_upstream_api,
    resolve_device,
)
from utils.dataset_sequences import DATASET_SEQUENCE, RETRIEVAL_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        required=True,
        help="Checkout of facebookresearch/meru or PalAvik/hycoclip",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Official training config; inferred from --upstream-repo when omitted",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-storage-base", type=Path, required=True)
    parser.add_argument("--datasets", default=DATASET_SEQUENCE)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_path(root: Path, dataset: str, split: str, modality: str) -> Path:
    return root / dataset / f"{dataset}_{split}_{modality}.h5"


def _write_h5(
    path: Path,
    features: torch.Tensor,
    ids: Iterable[int],
    *,
    dataset: str,
    split: str,
    modality: str,
    backbone: str,
    checkpoint_sha256: str,
    curvature: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids_tensor = torch.as_tensor(list(ids), dtype=torch.long)
    features = features.detach().cpu().float()
    if features.ndim != 2 or features.shape[0] != ids_tensor.numel():
        raise ValueError(
            f"Invalid {dataset}:{split}:{modality} shapes: "
            f"features={tuple(features.shape)}, ids={tuple(ids_tensor.shape)}"
        )

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "features", data=features.numpy(), compression="gzip", compression_opts=4
        )
        id_key = "ids" if dataset in RETRIEVAL_DATASETS else "labels"
        handle.create_dataset(id_key, data=ids_tensor.numpy(), compression="gzip")
        handle.attrs["dataset_name"] = dataset
        handle.attrs["split"] = split
        handle.attrs["feature_type"] = modality
        handle.attrs["backbone"] = backbone
        handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        handle.attrs["num_samples"] = features.shape[0]
        handle.attrs["feature_dim"] = features.shape[1]
        handle.attrs["curv"] = float(curvature)
        handle.attrs["projected_to_hyperboloid"] = True


def _existing_classification(root: Path, dataset: str, split: str) -> bool:
    return all(
        _feature_path(root, dataset, split, modality).exists()
        for modality in ("image", "text")
    )


def _existing_retrieval(root: Path, dataset: str, split: str) -> bool:
    gt_path = root / dataset / f"{dataset}_{split}_retrieval_gt.json"
    return _existing_classification(root, dataset, split) and gt_path.exists()


def _classification_transform(image_size: int):
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def _retrieval_transform(image_size: int):
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )


@torch.inference_mode()
def extract_classification(
    *,
    api,
    model: torch.nn.Module,
    dataset: str,
    split: str,
    data_root: Path,
    output_root: Path,
    backbone: str,
    checkpoint_sha256: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> None:
    source = api.dataset_catalog.build(
        dataset, data_root, split, _classification_transform(image_size)
    )
    loader = DataLoader(
        source,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )

    image_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for images, labels in tqdm(loader, desc=f"{dataset}:{split} images"):
        image_batches.append(model.encode_image(images.to(model.device), project=True).cpu())
        label_batches.append(torch.as_tensor(labels, dtype=torch.long).cpu())

    if not image_batches:
        raise RuntimeError(f"No samples found for {dataset}:{split}")
    image_features = torch.cat(image_batches)
    labels = torch.cat(label_batches)

    class_names = api.class_names.get(dataset)
    if class_names is None:
        raise KeyError(f"{dataset!r} is missing from {api.package}.evaluation.class_names")
    unique_labels = sorted(int(value) for value in labels.unique().tolist())
    tokenizer = api.tokenizer()
    per_class: dict[int, torch.Tensor] = {}
    for label in unique_labels:
        if label < 0 or label >= len(class_names):
            raise ValueError(
                f"Label {label} is outside the {len(class_names)} names for {dataset}"
            )
        tokens = tokenizer(class_names[label])
        per_class[label] = model.encode_text(tokens, project=True).squeeze(0).cpu()
    text_features = torch.stack([per_class[int(label)] for label in labels.tolist()])

    curvature = float(model.curv.exp().item()) if hasattr(model, "curv") else 1.0
    for modality, features in (("image", image_features), ("text", text_features)):
        _write_h5(
            _feature_path(output_root, dataset, split, modality),
            features,
            labels.tolist(),
            dataset=dataset,
            split=split,
            modality=modality,
            backbone=backbone,
            checkpoint_sha256=checkpoint_sha256,
            curvature=curvature,
        )


def _build_retrieval_dataset(api, dataset: str, split: str, root: Path, transform):
    if dataset == "coco":
        upstream_split = "train" if split == "train" else "val"
        return api.coco_captions(root / "coco", upstream_split, transform)
    if dataset == "flickr30k":
        upstream_split = "train" if split == "train" else "test"
        return api.flickr30k_captions(root / "flickr30k", upstream_split, transform)
    raise ValueError(f"Not a retrieval dataset: {dataset}")


@torch.inference_mode()
def extract_retrieval(
    *,
    api,
    model: torch.nn.Module,
    dataset: str,
    split: str,
    data_root: Path,
    output_root: Path,
    backbone: str,
    checkpoint_sha256: str,
    image_size: int,
) -> None:
    source = _build_retrieval_dataset(
        api, dataset, split, data_root, _retrieval_transform(image_size)
    )
    tokenizer = api.tokenizer()
    image_ids: list[int] = []
    text_ids: list[int] = []
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    image_to_text: defaultdict[int, set[int]] = defaultdict(set)
    text_to_image: defaultdict[int, set[int]] = defaultdict(set)

    for instance in tqdm(source, desc=f"{dataset}:{split} image-text"):
        image_id = int(instance["image_id"])
        caption_ids = [int(value) for value in instance["caption_ids"]]
        image_to_text[image_id].update(caption_ids)
        for caption_id in caption_ids:
            text_to_image[caption_id].add(image_id)

        image = instance["image"].unsqueeze(0).to(model.device)
        image_features.append(model.encode_image(image, project=True).cpu())
        caption_tokens = tokenizer(instance["captions"])
        text_features.append(model.encode_text(caption_tokens, project=True).cpu())
        image_ids.append(image_id)
        text_ids.extend(caption_ids)

    if not image_features or not text_features:
        raise RuntimeError(f"No samples found for {dataset}:{split}")
    image_tensor = torch.cat(image_features)
    text_tensor = torch.cat(text_features)
    curvature = float(model.curv.exp().item()) if hasattr(model, "curv") else 1.0

    _write_h5(
        _feature_path(output_root, dataset, split, "image"),
        image_tensor,
        image_ids,
        dataset=dataset,
        split=split,
        modality="image",
        backbone=backbone,
        checkpoint_sha256=checkpoint_sha256,
        curvature=curvature,
    )
    _write_h5(
        _feature_path(output_root, dataset, split, "text"),
        text_tensor,
        text_ids,
        dataset=dataset,
        split=split,
        modality="text",
        backbone=backbone,
        checkpoint_sha256=checkpoint_sha256,
        curvature=curvature,
    )
    gt_path = output_root / dataset / f"{dataset}_{split}_retrieval_gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "image_to_text_gt": {
                    str(key): sorted(values) for key, values in image_to_text.items()
                },
                "text_to_image_gt": {
                    str(key): sorted(values) for key, values in text_to_image.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    train_config = (
        args.train_config.resolve()
        if args.train_config is not None
        else default_train_config(args.backbone, args.upstream_repo)
    )
    if not train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {train_config}")

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    if not datasets or any(split not in {"train", "test"} for split in splits):
        raise ValueError("Provide datasets and splits chosen from: train,test")

    output_root = args.feature_storage_base.resolve() / args.backbone
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    api = load_upstream_api(args.backbone, args.upstream_repo)
    logger.info("Loading {} on {}", args.backbone, device)
    model = api.load_model(checkpoint, train_config, device)
    checkpoint_sha256 = _sha256(checkpoint)
    logger.info("Checkpoint SHA-256: {}", checkpoint_sha256)

    for dataset in datasets:
        for split in splits:
            if dataset in RETRIEVAL_DATASETS:
                if not args.force and _existing_retrieval(output_root, dataset, split):
                    logger.info("Skipping existing {}:{}", dataset, split)
                    continue
                extract_retrieval(
                    api=api,
                    model=model,
                    dataset=dataset,
                    split=split,
                    data_root=args.data_root.resolve(),
                    output_root=output_root,
                    backbone=args.backbone,
                    checkpoint_sha256=checkpoint_sha256,
                    image_size=args.image_size,
                )
            else:
                if not args.force and _existing_classification(output_root, dataset, split):
                    logger.info("Skipping existing {}:{}", dataset, split)
                    continue
                extract_classification(
                    api=api,
                    model=model,
                    dataset=dataset,
                    split=split,
                    data_root=args.data_root.resolve(),
                    output_root=output_root,
                    backbone=args.backbone,
                    checkpoint_sha256=checkpoint_sha256,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    image_size=args.image_size,
                )

    logger.info("Frozen features are ready at {}", output_root)


if __name__ == "__main__":
    main()
