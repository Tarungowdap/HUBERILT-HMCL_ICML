#!/usr/bin/env python3
"""Download/cache classification datasets and validate retrieval dataset layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.upstream import load_upstream_api
from utils.dataset_sequences import DATASET_SEQUENCE, RETRIEVAL_DATASETS


def _retrieval_requirements(root: Path) -> dict[str, list[Path]]:
    return {
        "coco": [
            root / "coco" / "train2017",
            root / "coco" / "val2017",
            root / "coco" / "annotations" / "captions_train2017.json",
            root / "coco" / "annotations" / "captions_val2017.json",
        ],
        "flickr30k": [
            root / "flickr30k" / "flickr30k_images",
            root / "flickr30k" / "dataset_flickr30k.json",
        ],
    }


def _referenced_retrieval_images(root: Path, dataset: str) -> list[Path]:
    if dataset == "coco":
        paths: list[Path] = []
        for split in ("train", "val"):
            annotation = root / "coco" / "annotations" / f"captions_{split}2017.json"
            payload = json.loads(annotation.read_text())
            paths.extend(
                root / "coco" / f"{split}2017" / instance["file_name"]
                for instance in payload["images"]
            )
        return paths
    if dataset == "flickr30k":
        annotation = root / "flickr30k" / "dataset_flickr30k.json"
        payload = json.loads(annotation.read_text())
        return [
            root / "flickr30k" / "flickr30k_images" / instance["filename"]
            for instance in payload["images"]
        ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default=DATASET_SEQUENCE)
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Only validate manually downloaded COCO/Flickr30K",
    )
    parser.add_argument(
        "--skip-retrieval-check",
        action="store_true",
        help="Prepare classification datasets without requiring COCO/Flickr30K",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    api = load_upstream_api("meru_vit_b", args.upstream_repo)

    if not args.skip_classification:
        for dataset in datasets:
            if dataset in RETRIEVAL_DATASETS:
                continue
            for split in ("train", "test"):
                source = api.dataset_catalog.build(dataset, root, split, transform=None)
                try:
                    size = len(source)
                except TypeError:
                    size = "iterable"
                print(f"ready: {dataset}:{split} ({size})")

    missing: list[Path] = []
    requirements = _retrieval_requirements(root)
    if not args.skip_retrieval_check:
        for dataset in datasets:
            for path in requirements.get(dataset, []):
                if not path.exists():
                    missing.append(path)
                else:
                    print(f"ready: {path}")
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Retrieval data are not auto-downloaded because their licenses/access "
            f"terms require manual preparation. Missing:\n{formatted}\nSee README.md."
        )

    if not args.skip_retrieval_check:
        for dataset in datasets:
            if dataset not in RETRIEVAL_DATASETS:
                continue
            referenced = _referenced_retrieval_images(root, dataset)
            missing_images = [path for path in referenced if not path.is_file()]
            if missing_images:
                examples = "\n".join(f"  - {path}" for path in missing_images[:10])
                remaining = len(missing_images) - min(len(missing_images), 10)
                suffix = f"\n  ... and {remaining} more" if remaining else ""
                raise FileNotFoundError(
                    f"{dataset} annotations reference {len(missing_images)} missing "
                    f"image file(s):\n{examples}{suffix}"
                )
            print(f"ready: {dataset} ({len(referenced)} referenced images)")


if __name__ == "__main__":
    main()
