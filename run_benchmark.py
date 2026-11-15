#!/usr/bin/env python3
"""Command-line benchmark: Split-CIFAR100 / Split-MiniImageNet, all methods.

Example:
    python run_benchmark.py --dataset split_cifar100 --seeds 0 1 2 --amp 1

Methods (identical encoder, data stream, optimiser and evaluation for all):
  euclidean             - Euclidean head, no forgetting protection (lower bound).
  ewc                   - Euclidean head + diagonal-Fisher EWC (Kirkpatrick et al., 2017).
  hmcl                  - Fixed-curvature Lorentz head + covariance gradient projection + head
                          pullback (HMCL-style, adapted to end-to-end training).
  dynamic-curvature  - OURS: learnable-curvature hyperbolic prototypes + Fisher protection
                          + prototype drift compensation + old prototypes as negatives with
                          prototype augmentation. Zero replay.
  ablation-euclidean          - ablation: our recipe with a Euclidean head (does geometry matter?)
  ablation-fixed-curvature    - ablation: our recipe with c frozen (does dynamic curvature matter?)

Datasets (downloaded automatically from Kaggle via kagglehub, zero replay):
  split_cifar100       fedesoriano/cifar100 (CIFAR-100 python pickles), 32x32
  split_miniimagenet   arjunashok33/miniimagenet (100 class folders x 600 images), resized to --img-size
  Any other Kaggle dataset can be used with --kaggle-dataset owner/name; the loader detects
  CIFAR pickles, train/ + test|val/ class folders, or a single pool of class folders
  (then 1/6 of each class is held out for testing). --source torchvision-hf uses the old loaders.

The encoder is a small ResNet trained from scratch (no pretrained weights), so it reads
raw images and its input size is fixed by the data, not hard-coded.

Two evaluation protocols are reported:
  task-IL  : the task id is known, prediction among that task's classes only.
  class-IL : no task id, prediction among ALL classes seen so far (harder, standard).
"""

from __future__ import annotations

import json

import torch

from hmcl_cl import METHOD_ALIASES, GPUAugment, build_parser, build_stream, print_table, run_experiment


def main() -> None:
    args = build_parser().parse_args()

    methods = [METHOD_ALIASES[m.lower()] for m in args.methods]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.dataset == "split_cifar100":
        aug = GPUAugment((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    else:
        aug = GPUAugment((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), pad=args.img_size // 8)

    results: dict[str, list[dict]] = {m: [] for m in methods}
    for seed in args.seeds:
        stream = build_stream(args.dataset, args.data_root, args.tasks, args.classes_per_task,
                              args.img_size, seed, device, args.source, args.kaggle_dataset)
        for tn, d in stream.items():
            for split, cap in (("train", args.max_train_per_task), ("test", args.max_test_per_task)):
                if cap and len(d[f"{split}_y"]) > cap:
                    keep = torch.randperm(len(d[f"{split}_y"]), generator=torch.Generator().manual_seed(seed))[:cap]
                    keep = keep.to(device)
                    d[f"{split}_x"], d[f"{split}_y"] = d[f"{split}_x"][keep], d[f"{split}_y"][keep]
        for m in methods:
            results[m].append(run_experiment(m, stream, args, seed, device, aug))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({"config": {k: str(v) for k, v in vars(args).items()}, "results": results}, f, indent=2)
        del stream
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_table(results)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
