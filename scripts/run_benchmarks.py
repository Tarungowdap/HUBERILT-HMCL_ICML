"""Unified benchmark script comparing Euclidean, HMCL (fixed curvature), Dynamic-Curvature, and EWC baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger


def run_single_method(
    method: str,
    feature_dir: Path,
    datasets: str,
    output_path: Path,
    backbone: str = "meru_vit_b",
    seed: int = 42,
    device: str = "cpu",
    cls_epochs: int = 1,
    cls_batch_size: int = 64,
    ret_epochs: int = 1,
    ret_batch_size: int = 64,
    curv_lr: float = 1e-4,
    curv_reg: float = 1.0,
    ewc_lambda: float = 100.0,
    ewc_gamma: float = 1.0,
) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    cmd = [
        sys.executable,
        str(main_py),
        "--feature-storage-dir",
        str(feature_dir),
        "--datasets",
        datasets,
        "--backbone",
        backbone,
        "--seed",
        str(seed),
        "--device",
        device,
        "--method",
        method,
        "--output",
        str(output_path),
        "--cls-epochs",
        str(cls_epochs),
        "--cls-batch-size",
        str(cls_batch_size),
        "--ret-epochs",
        str(ret_epochs),
        "--ret-batch-size",
        str(ret_batch_size),
    ]

    if method == "dynamic-curvature":
        cmd += [
            "--curvature-min",
            "0.05",
            "--curvature-max",
            "0.5",
            "--curvature-lr",
            str(curv_lr),
            "--curvature-reg",
            str(curv_reg),
        ]
    elif method == "ewc":
        cmd += [
            "--ewc-lambda",
            str(ewc_lambda),
            "--ewc-gamma",
            str(ewc_gamma),
        ]

    logger.info("Executing benchmark for method: {}...", method)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    res = subprocess.run(cmd, cwd=str(repo_root), env=env)
    if res.returncode != 0:
        logger.error("Command failed: {}", " ".join(cmd))
        raise RuntimeError(f"Method {method} failed with code {res.returncode}")

    with open(output_path, "r") as f:
        return json.load(f)


def print_comparison_table(results_dict: dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'CONTINUAL LEARNING BENCHMARK COMPARISON':^90}")
    print("=" * 90)
    methods = list(results_dict.keys())
    header = f"{'Metric':<30}" + "".join(f" | {m.upper():<16}" for m in methods)
    print(header)
    print("-" * len(header))

    metrics = [
        ("Final Overall Acc (%)", ("overall", "final")),
        ("Overall Forgetting (%)", ("overall", "forgetting")),
        ("Signed BWT (%)", ("overall", "signed_bwt")),
        ("Classification Acc (%)", ("classification", "final_acc")),
        ("Classification Forgetting", ("classification", "forgetting")),
        ("Mean Representation Drift", ("drift", "mean_drift")),
        ("Curvature at End", ("curvature_history", -1)),
    ]

    for label, path in metrics:
        row = f"{label:<30}"
        for m in methods:
            data = results_dict.get(m, {})
            val = data
            for k in path:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                elif isinstance(val, list) and isinstance(k, int) and -len(val) <= k < len(val):
                    val = val[k]
                else:
                    val = None
                    break
            if val is not None and isinstance(val, (int, float)):
                row += f" | {val:<16.4f}"
            elif val is not None:
                row += f" | {str(val):<16}"
            else:
                row += f" | {'N/A':<16}"
        print(row)

    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run side-by-side benchmark comparison")
    parser.add_argument(
        "--feature-storage-dir",
        type=Path,
        default=Path("feature_storage/meru_vit_b"),
    )
    parser.add_argument("--datasets", default="aircraft,caltech101,cifar10")
    parser.add_argument("--backbone", default="meru_vit_b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cls-epochs", type=int, default=2)
    parser.add_argument("--cls-batch-size", type=int, default=64)
    parser.add_argument("--ret-epochs", type=int, default=2)
    parser.add_argument("--ret-batch-size", type=int, default=64)
    parser.add_argument(
        "--methods",
        default="euclidean,ewc,hmcl,dynamic-curvature",
        help="Comma-separated methods to benchmark",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    args.results_dir = (repo_root / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir
    args.feature_storage_dir = (repo_root / args.feature_storage_dir).resolve() if not args.feature_storage_dir.is_absolute() else args.feature_storage_dir
    args.results_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results_summary: dict[str, dict] = {}

    for method in methods:
        out_path = args.results_dir / f"{method}_seed{args.seed}.json"
        res = run_single_method(
            method=method,
            feature_dir=args.feature_storage_dir,
            datasets=args.datasets,
            output_path=out_path,
            backbone=args.backbone,
            seed=args.seed,
            device=args.device,
            cls_epochs=args.cls_epochs,
            cls_batch_size=args.cls_batch_size,
            ret_epochs=args.ret_epochs,
            ret_batch_size=args.ret_batch_size,
        )
        results_summary[method] = res

    summary_file = args.results_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Saved complete benchmark summary to {}", summary_file)

    print_comparison_table(results_summary)


if __name__ == "__main__":
    main()
