#!/usr/bin/env python3
"""Load an official checkpoint and run one image/text encoding smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils.upstream import (
    BACKBONES,
    default_train_config,
    load_upstream_api,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    train_config = (
        args.train_config.resolve()
        if args.train_config is not None
        else default_train_config(args.backbone, args.upstream_repo)
    )
    if not train_config.is_file():
        raise FileNotFoundError(train_config)

    device = resolve_device(args.device)
    api = load_upstream_api(args.backbone, args.upstream_repo)
    model = api.load_model(checkpoint, train_config, device)
    tokenizer = api.tokenizer()

    with torch.inference_mode():
        image = torch.zeros(1, 3, 224, 224, device=device)
        image_features = model.encode_image(image, project=True)
        text_features = model.encode_text(tokenizer("a test image"), project=True)

    if image_features.ndim != 2 or image_features.shape != text_features.shape:
        raise ValueError(
            f"Image/text output mismatch: {image_features.shape} vs {text_features.shape}"
        )
    if not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all():
        raise ValueError("The model emitted non-finite features")

    curvature = float(model.curv.exp().item()) if hasattr(model, "curv") else 1.0
    print(
        f"valid: backbone={args.backbone}, device={device}, "
        f"feature_dim={image_features.shape[1]}, curvature={curvature:.8g}"
    )


if __name__ == "__main__":
    main()
