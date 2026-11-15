"""Run continual learning experiments across 5 seeds and compute mean and std."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5 seeds and summarize results")
    parser.add_argument("backbone", nargs="?", default="meru_vit_b")
    parser.add_argument("--features-root", type=Path, default=Path("feature_storage/meru_vit_b"))
    parser.add_argument("--method", default="dynamic-curvature", choices=["hmcl", "dynamic-curvature", "ewc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--datasets", default="aircraft,caltech101,coco")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cls-epochs", type=int, default=2)
    parser.add_argument("--ret-epochs", type=int, default=2)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "main.py"
    results_dir = repo_root / "results" / f"{args.method}_{args.backbone}_5seeds"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)

    for seed in args.seeds:
        out_file = results_dir / f"seed_{seed}.json"
        cmd = [
            sys.executable,
            str(main_py),
            "--feature-storage-dir",
            str(args.features_root),
            "--backbone",
            args.backbone,
            "--datasets",
            args.datasets,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--method",
            args.method,
            "--cls-epochs",
            str(args.cls_epochs),
            "--ret-epochs",
            str(args.ret_epochs),
            "--output",
            str(out_file),
        ]
        logger.info("Running seed {}...", seed)
        subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)
        with open(out_file, "r") as f:
            metrics_list.append(json.load(f))

    finals = [m["overall"]["final"] for m in metrics_list]
    forgettings = [m["overall"]["forgetting"] for m in metrics_list]
    bwts = [m["overall"]["signed_bwt"] for m in metrics_list]

    summary = {
        "method": args.method,
        "backbone": args.backbone,
        "seeds": args.seeds,
        "overall_final_mean": float(np.mean(finals)),
        "overall_final_std": float(np.std(finals)),
        "forgetting_mean": float(np.mean(forgettings)),
        "forgetting_std": float(np.std(forgettings)),
        "signed_bwt_mean": float(np.mean(bwts)),
        "signed_bwt_std": float(np.std(bwts)),
    }

    summary_file = results_dir / "five_seed_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("5-seed Summary:\n{}", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
