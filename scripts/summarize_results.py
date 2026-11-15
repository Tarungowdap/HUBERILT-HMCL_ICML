#!/usr/bin/env python3
"""Aggregate HMCL seed results using sample standard deviation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = {
    "performance": ("overall", "final"),
    "overall_forgetting": ("overall", "forgetting"),
    "overall_bwt": ("overall", "signed_bwt"),
    "classification_acc": ("classification", "final_acc"),
    "classification_forgetting": ("classification", "forgetting"),
    "classification_bwt": ("classification", "signed_bwt"),
    "retrieval_i2t_rk": ("retrieval", "final_i2t_rk"),
    "retrieval_forgetting": (
        "retrieval",
        "forgetting_i2t_rk",
    ),
    "retrieval_bwt": (
        "retrieval",
        "signed_bwt_i2t_rk",
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [json.loads(path.read_text()) for path in args.results]
    if len(runs) < 2:
        raise ValueError("At least two seed results are required")

    backbone = runs[0]["config"]["backbone"]
    datasets = runs[0]["datasets"]
    reference_config = {
        "variance_ratio": runs[0]["config"]["variance_ratio"],
        "pullback": runs[0]["config"]["pullback"],
        "reported_retrieval": runs[0]["config"].get("reported_retrieval"),
    }
    for run in runs[1:]:
        if run["config"]["backbone"] != backbone or run["datasets"] != datasets:
            raise ValueError("All results must use the same backbone and task sequence")
        run_config = {
            "variance_ratio": run["config"]["variance_ratio"],
            "pullback": run["config"]["pullback"],
            "reported_retrieval": run["config"].get("reported_retrieval"),
        }
        if run_config != reference_config:
            raise ValueError(
                "All results must use the same variance ratio, pullback, and "
                "reported retrieval metric"
            )

    summary = {
        "method": "HMCL",
        "backbone": backbone,
        "seeds": [run["config"]["seed"] for run in runs],
        "num_runs": len(runs),
        "config": reference_config,
        "metrics": {},
    }
    for name, path in METRICS.items():
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(float(value))
        mean = statistics.mean(values)
        variance = statistics.variance(values)
        std = statistics.stdev(values)
        summary["metrics"][name] = {
            "mean": mean,
            "sample_variance": variance,
            "sample_std": std,
            "formatted": f"{mean:.2f}±{std:.2f}",
            "values": values,
        }

    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
