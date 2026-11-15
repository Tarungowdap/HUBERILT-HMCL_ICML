#!/usr/bin/env python3
"""Validate a complete frozen-feature store before running HMCL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset_sequences import DATASET_SEQUENCE, RETRIEVAL_DATASETS


def _read(path: Path) -> tuple[tuple[int, int], list[int], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features' dataset in {path}")
        id_key = "labels" if "labels" in handle else "ids"
        if id_key not in handle:
            raise KeyError(f"Missing 'labels' or 'ids' dataset in {path}")
        features = handle["features"]
        ids = [int(value) for value in handle[id_key][:]]
        attrs = {key: handle.attrs[key] for key in handle.attrs}
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError(f"Expected a non-empty 2-D feature array in {path}")
        if features.shape[0] != len(ids):
            raise ValueError(f"Feature/ID count mismatch in {path}")

        for start in range(0, features.shape[0], 8192):
            if not np.isfinite(features[start : start + 8192]).all():
                raise ValueError(f"Non-finite feature value in {path}")
        shape = (int(features.shape[0]), int(features.shape[1]))
    return shape, ids, attrs


def _gt(path: Path) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    image_to_text = {
        int(key): {int(value) for value in values}
        for key, values in payload.get("image_to_text_gt", {}).items()
    }
    text_to_image = {
        int(key): {int(value) for value in values}
        for key, values in payload.get("text_to_image_gt", {}).items()
    }
    if not image_to_text or not text_to_image:
        raise ValueError(f"Empty retrieval ground truth in {path}")
    return image_to_text, text_to_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--datasets", default=DATASET_SEQUENCE)
    parser.add_argument("--splits", default="train,test")
    args = parser.parse_args()

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not datasets or not splits or any(value not in {"train", "test"} for value in splits):
        raise ValueError("Provide non-empty datasets and splits chosen from train,test")

    feature_dims: set[int] = set()
    checkpoint_hashes: set[str] = set()
    file_count = 0
    sample_count = 0

    for dataset in datasets:
        for split in splits:
            records = {}
            for modality in ("image", "text"):
                path = args.feature_root / dataset / f"{dataset}_{split}_{modality}.h5"
                shape, ids, attrs = _read(path)
                if "backbone" in attrs and str(attrs["backbone"]) != args.backbone:
                    raise ValueError(
                        f"Backbone mismatch in {path}: {attrs['backbone']!r} != {args.backbone!r}"
                    )
                feature_dims.add(shape[1])
                if "checkpoint_sha256" in attrs:
                    checkpoint_hashes.add(str(attrs["checkpoint_sha256"]))
                records[modality] = (shape, ids)
                file_count += 1
                sample_count += shape[0]

            image_shape, image_ids = records["image"]
            text_shape, text_ids = records["text"]
            if image_shape[1] != text_shape[1]:
                raise ValueError(f"Image/text dimensions differ for {dataset}:{split}")

            if dataset not in RETRIEVAL_DATASETS:
                if image_ids != text_ids or image_shape[0] != text_shape[0]:
                    raise ValueError(f"Classification alignment failed for {dataset}:{split}")
                continue

            gt_path = args.feature_root / dataset / f"{dataset}_{split}_retrieval_gt.json"
            image_to_text, text_to_image = _gt(gt_path)
            if not set(image_ids).issubset(image_to_text):
                raise ValueError(f"Missing image IDs in {gt_path}")
            if not set(text_ids).issubset(text_to_image):
                raise ValueError(f"Missing text IDs in {gt_path}")

    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions: {sorted(feature_dims)}")
    if len(checkpoint_hashes) > 1:
        raise ValueError("More than one checkpoint SHA-256 appears in the feature store")

    print(
        f"valid: backbone={args.backbone}, datasets={len(datasets)}, "
        f"files={file_count}, stored_rows={sample_count}, feature_dim={feature_dims.pop()}"
    )
    if checkpoint_hashes:
        print(f"checkpoint_sha256={checkpoint_hashes.pop()}")


if __name__ == "__main__":
    main()
