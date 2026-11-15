#!/usr/bin/env python3
"""Ablation study: compare all 4 continual learning methods side by side.

Runs: euclidean, hmcl, dynamic-curvature, ewc
Prints a comparison table with: Final Acc, Forgetting, BWT, Retrieval R@5, Drift metrics.

Usage:
  python scripts/ablation_study.py \\
    --feature-storage-dir feature_storage/meru_vit_b \\
    --datasets aircraft,caltech101,coco \\
    --cls-epochs 2 --ret-epochs 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


METHODS = [
    ("euclidean",         []),
    ("hmcl",              []),
    ("dynamic-curvature", ["--curvature-min", "0.05", "--curvature-max", "0.5", "--curvature-lr", "1e-4", "--curvature-reg", "1.0"]),
    ("ewc",               ["--ewc-lambda", "100", "--ewc-gamma", "1.0"]),
]


def run_method(
    method: str,
    extra_args: list[str],
    base_args: list[str],
    results_dir: Path,
) -> dict:
    out_path = results_dir / f"{method}.json"
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--method", method,
        "--output", str(out_path),
    ] + base_args + extra_args

    print(f"\n{'='*60}")
    print(f"  Running: {method.upper()}")
    print(f"{'='*60}")
    print("  CMD:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT), env={
            **__import__("os").environ,
            "PYTHONPATH": str(ROOT),
        })
        with open(out_path) as f:
            return json.load(f)
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: {method} failed with code {exc.returncode}")
        return {}
    except FileNotFoundError:
        print(f"  ERROR: results file not written for {method}")
        return {}


def print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print a formatted comparison table to terminal."""
    methods = list(all_results.keys())
    if not methods:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Method':<20} {'Cls Acc':>10} {'Cls Forg':>10} {'BWT':>8} {'Ret R@5':>10} {'Drift(mean)':>14} {'Curv@end':>10}")
    print("-" * 90)

    for method, res in all_results.items():
        cls = res.get("classification", {})
        ret = res.get("retrieval", {})
        drift = res.get("drift", {})
        curv_hist = res.get("curvature_history", [])

        cls_acc  = cls.get("final_acc", float("nan"))
        cls_forg = cls.get("forgetting", float("nan"))
        bwt      = cls.get("signed_bwt", float("nan"))
        ret_r5   = ret.get("final_i2t_rk", float("nan"))
        mean_drf = drift.get("mean_drift", float("nan"))
        final_c  = curv_hist[-1] if curv_hist else float("nan")

        print(
            f"  {method:<20} {cls_acc:>10.2f} {cls_forg:>10.2f} {bwt:>8.2f} "
            f"{ret_r5:>10.2f} {mean_drf:>14.6f} {final_c:>10.4f}"
        )
    print("=" * 90)

    print("\n  REPRESENTATION DRIFT BREAKDOWN (Task-1 embeddings: initial → final)")
    print("-" * 70)
    print(f"  {'Method':<20} {'mean':>10} {'max':>10} {'radial':>10} {'angular':>10}")
    print("-" * 70)
    for method, res in all_results.items():
        d = res.get("drift", {})
        print(
            f"  {method:<20}"
            f"  {d.get('mean_drift', float('nan')):>10.6f}"
            f"  {d.get('max_drift', float('nan')):>10.6f}"
            f"  {d.get('radial_drift', float('nan')):>10.6f}"
            f"  {d.get('angular_drift', float('nan')):>10.6f}"
        )
    print("-" * 70)

    if "dynamic-curvature" in all_results:
        hist = all_results["dynamic-curvature"].get("curvature_history", [])
        if hist:
            print(f"\n  CURVATURE TRAJECTORY (dynamic-curvature):")
            for i, c in enumerate(hist):
                print(f"    After task {i+1}: c={c:.4f}")

    print("\n  KEY:")
    print("  Cls Acc  = Final mean classification accuracy (%)")
    print("  Cls Forg = Mean forgetting on classification tasks (%)")
    print("  BWT      = Backward Transfer (positive = no forgetting)")
    print("  Ret R@5  = Final Image-to-Text R@5 on retrieval tasks (%)")
    print("  Drift    = Representation drift of Task-1 embeddings after all tasks")
    print("  Curv@end = Curvature at end of training (nan = Euclidean)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study: compare all CL methods")
    parser.add_argument("--feature-storage-dir", required=True)
    parser.add_argument("--backbone", default="meru_vit_b",
                        choices=("meru_vit_l", "meru_vit_b", "hycoclip_vit_b"))
    parser.add_argument("--datasets", default="aircraft,caltech101,coco")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--cls-epochs", type=int, default=2)
    parser.add_argument("--ret-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ablation")
    parser.add_argument("--methods", default="euclidean,hmcl,dynamic-curvature,ewc",
                        help="Comma-separated subset of methods to run")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--feature-storage-dir", args.feature_storage_dir,
        "--backbone", args.backbone,
        "--datasets", args.datasets,
        "--device", args.device,
        "--cls-epochs", str(args.cls_epochs),
        "--ret-epochs", str(args.ret_epochs),
        "--seed", str(args.seed),
    ]

    requested_methods = {m.strip() for m in args.methods.split(",")}
    all_results: dict[str, dict] = {}

    for method, extra in METHODS:
        if method not in requested_methods:
            continue
        res = run_method(method, extra, base_args, args.results_dir)
        if res:
            all_results[method] = res

    summary_path = args.results_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved summary to: {summary_path}")

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
