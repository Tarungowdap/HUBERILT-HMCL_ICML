#!/usr/bin/env python3
"""Train and evaluate continual learning methods on a multimodal task stream.

Supported methods
-----------------
  hmcl              Fixed-curvature hyperbolic (official HMCL baseline)
  dynamic-curvature Hyperbolic with learnable, bounded curvature per task
  euclidean         Euclidean cosine-similarity baseline
  ewc               Euclidean + Elastic Weight Consolidation

Usage example
-------------
  python main.py \\
    --feature-storage-dir feature_storage/meru_vit_b \\
    --backbone meru_vit_b \\
    --datasets aircraft,caltech101,coco \\
    --method dynamic-curvature \\
    --cls-epochs 5 --ret-epochs 5 \\
    --output results/dyn_curv.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from data_utils.features import FeatureStore
from trainer.trainer import ContinualTrainer
from utils.config import ClassificationConfig, RetrievalConfig
from utils.dataset_sequences import DATASET_SEQUENCE, is_retrieval_dataset
from utils.drift_metrics import compute_representation_drift


RETRIEVAL_METRICS = (
    "t2i_r1",
    "t2i_r5",
    "t2i_r10",
    "i2t_r1",
    "i2t_r5",
    "i2t_r10",
)

DEFAULT_PULLBACK_BETA = {
    "meru_vit_l": 0.10,
    "meru_vit_b": 0.10,
    "hycoclip_vit_b": 0.20,
}

REPORTED_RETRIEVAL_K = {
    "meru_vit_l": 5,
    "meru_vit_b": 5,
    "hycoclip_vit_b": 1,
}

DEFAULT_VARIANCE_RATIO = {
    "meru_vit_l": 0.9,
    "meru_vit_b": 0.8,
    "hycoclip_vit_b": 0.9,
}

CLASSIFICATION_DEFAULTS = ClassificationConfig()
RETRIEVAL_DEFAULTS = RetrievalConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperbolic Continual Learning: HMCL / Dynamic-Curvature / Euclidean / EWC"
    )
    parser.add_argument(
        "--datasets", default=DATASET_SEQUENCE, help="Comma-separated task sequence"
    )
    parser.add_argument(
        "--feature-storage-dir", required=True, help="Root dir with cached HDF5 features"
    )
    parser.add_argument(
        "--backbone",
        default="meru_vit_b",
        choices=("meru_vit_l", "meru_vit_b", "hycoclip_vit_b"),
    )
    parser.add_argument(
        "--device",
        default=CLASSIFICATION_DEFAULTS.device,
        choices=("auto", "cuda", "cpu"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None, help="Optional result JSON path")

    parser.add_argument(
        "--method",
        default="hmcl",
        choices=("hmcl", "dynamic-curvature", "euclidean", "ewc"),
        help=(
            "Continual learning method: "
            "hmcl=Fixed hyperbolic baseline, "
            "dynamic-curvature=Learnable curvature (ours), "
            "euclidean=Cosine-similarity baseline, "
            "ewc=Euclidean+ElasticWeightConsolidation"
        ),
    )

    parser.add_argument("--curv", type=float, default=CLASSIFICATION_DEFAULTS.curv,
                        help="Initial/fixed curvature for hmcl and dynamic-curvature")
    parser.add_argument("--curvature-min", type=float, default=0.05,
                        help="Min curvature for dynamic-curvature method")
    parser.add_argument("--curvature-max", type=float, default=0.5,
                        help="Max curvature for dynamic-curvature method")
    parser.add_argument("--curvature-lr", type=float, default=1e-4,
                        help="Learning rate for curvature parameter")
    parser.add_argument("--curvature-reg", type=float, default=1.0,
                        help="Anchor regularisation weight for curvature drift")

    parser.add_argument("--ewc-lambda", type=float, default=100.0,
                        help="EWC regularisation strength")
    parser.add_argument("--ewc-gamma", type=float, default=1.0,
                        help="EWC Fisher decay factor (1.0 = accumulate all tasks)")

    parser.add_argument(
        "--variance-ratio", type=float, default=None,
        help="PCA variance ratio (default: reported value for selected backbone)"
    )
    parser.add_argument("--cls-lr", type=float, default=CLASSIFICATION_DEFAULTS.lr)
    parser.add_argument("--cls-weight-decay", type=float,
                        default=CLASSIFICATION_DEFAULTS.weight_decay)
    parser.add_argument("--cls-epochs", type=int, default=CLASSIFICATION_DEFAULTS.epochs)
    parser.add_argument("--cls-batch-size", type=int,
                        default=CLASSIFICATION_DEFAULTS.batch_size)
    parser.add_argument("--cls-entail-weight", type=float,
                        default=CLASSIFICATION_DEFAULTS.entail_weight)
    parser.add_argument("--ret-lr", type=float, default=RETRIEVAL_DEFAULTS.lr)
    parser.add_argument("--ret-weight-decay", type=float,
                        default=RETRIEVAL_DEFAULTS.weight_decay)
    parser.add_argument("--ret-epochs", type=int, default=RETRIEVAL_DEFAULTS.epochs)
    parser.add_argument("--ret-batch-size", type=int,
                        default=RETRIEVAL_DEFAULTS.batch_size)
    parser.add_argument("--ret-entail-weight", type=float,
                        default=RETRIEVAL_DEFAULTS.entail_weight)
    parser.add_argument(
        "--pullback-beta", type=float, default=None,
        help="Pullback coefficient (default: 0.10 for MERU, 0.20 for HyCoCLIP)"
    )
    return parser.parse_args()


def _resolve_pullback_beta(backbone: str, requested_beta: float | None) -> float:
    if requested_beta is not None:
        return float(requested_beta)
    return DEFAULT_PULLBACK_BETA[backbone]


def _resolve_variance_ratio(backbone: str, requested_ratio: float | None) -> float:
    if requested_ratio is not None:
        return float(requested_ratio)
    return DEFAULT_VARIANCE_RATIO[backbone]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(
    datasets: list[str],
    score_matrix: list[list[float]],
    retrieval_matrices: dict[str, list[list[float]]],
    reported_retrieval_k: int,
) -> dict[str, object]:
    task_count = len(score_matrix)
    final_scores = score_matrix[-1]
    forgetting = []
    bwt = []
    for task_index in range(task_count):
        final = final_scores[task_index]
        best = max(row[task_index] for row in score_matrix[task_index:])
        immediate = score_matrix[task_index][task_index]
        forgetting.append(max(0.0, best - final))
        bwt.append(immediate - final)

    classification_indices = [
        i for i, name in enumerate(datasets) if not is_retrieval_dataset(name)
    ]
    retrieval_indices = [
        i for i, name in enumerate(datasets) if is_retrieval_dataset(name)
    ]

    classification = {
        "final_acc": _mean([final_scores[i] for i in classification_indices]),
        "forgetting": _mean([forgetting[i] for i in classification_indices]),
        "signed_bwt": -_mean([bwt[i] for i in classification_indices]),
    }

    retrieval_detail: dict[str, dict[str, float]] = {}
    for metric in RETRIEVAL_METRICS:
        final_values = retrieval_matrices[metric][-1]
        metric_forgetting = []
        metric_bwt = []
        for retrieval_position, task_index in enumerate(retrieval_indices):
            final = final_values[retrieval_position]
            history = [
                retrieval_matrices[metric][step][retrieval_position]
                for step in range(task_index, task_count)
                if retrieval_position < len(retrieval_matrices[metric][step])
            ]
            immediate = retrieval_matrices[metric][task_index][retrieval_position]
            metric_forgetting.append(max(0.0, max(history) - final))
            metric_bwt.append(immediate - final)
        retrieval_detail[metric] = {
            "final": _mean(final_values),
            "forgetting": _mean(metric_forgetting),
            "signed_bwt": -_mean(metric_bwt),
        }

    reported_metric = f"i2t_r{reported_retrieval_k}"
    retrieval = {
        "reported_metric": reported_metric,
        "reported_k": reported_retrieval_k,
        "final_i2t_rk": retrieval_detail[reported_metric]["final"],
        "forgetting_i2t_rk": retrieval_detail[reported_metric]["forgetting"],
        "signed_bwt_i2t_rk": retrieval_detail[reported_metric]["signed_bwt"],
        "detail": retrieval_detail,
    }
    return {
        "overall": {
            "final": _mean(final_scores),
            "forgetting": _mean(forgetting),
            "signed_bwt": -_mean(bwt),
        },
        "classification": classification,
        "retrieval": retrieval,
        "score_matrix": score_matrix,
        "retrieval_matrices": retrieval_matrices,
    }


def _print_benchmark_table(results: dict, method: str) -> None:
    """Print a formatted benchmark table to terminal."""
    print("\n" + "=" * 60)
    print(f"  RESULTS: {method.upper()}")
    print("=" * 60)
    cls = results.get("classification", {})
    ret = results.get("retrieval", {})
    overall = results.get("overall", {})
    print(f"  Classification:")
    print(f"    Final Accuracy : {cls.get('final_acc', 0):.2f}%")
    print(f"    Forgetting     : {cls.get('forgetting', 0):.2f}%")
    print(f"    Signed BWT     : {cls.get('signed_bwt', 0):.2f}%")
    print(f"  Retrieval:")
    print(f"    {ret.get('reported_metric','i2t_r5')} Final: {ret.get('final_i2t_rk', 0):.2f}%")
    print(f"    Forgetting     : {ret.get('forgetting_i2t_rk', 0):.2f}%")
    print(f"  Overall:")
    print(f"    Final Avg      : {overall.get('final', 0):.2f}%")
    print(f"    Avg Forgetting : {overall.get('forgetting', 0):.2f}%")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    pullback_beta = _resolve_pullback_beta(args.backbone, args.pullback_beta)
    variance_ratio = _resolve_variance_ratio(args.backbone, args.variance_ratio)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("At least one dataset is required")

    classification_cfg = ClassificationConfig(
        curv=args.curv,
        device=args.device,
        lr=args.cls_lr,
        weight_decay=args.cls_weight_decay,
        epochs=args.cls_epochs,
        batch_size=args.cls_batch_size,
        entail_weight=args.cls_entail_weight,
    )
    retrieval_cfg = RetrievalConfig(
        curv=args.curv,
        device=args.device,
        lr=args.ret_lr,
        weight_decay=args.ret_weight_decay,
        epochs=args.ret_epochs,
        batch_size=args.ret_batch_size,
        entail_weight=args.ret_entail_weight,
    )

    logger.info(
        "Method={} | backbone={} | datasets={} | curvature={} | seed={}",
        args.method,
        args.backbone,
        datasets,
        args.curv,
        args.seed,
    )

    trainer = ContinualTrainer(
        classification_cfg,
        retrieval_cfg,
        variance_ratio=variance_ratio,
        pullback_beta=pullback_beta,
        method=args.method,
        curvature_min=args.curvature_min,
        curvature_max=args.curvature_max,
        curvature_lr=args.curvature_lr,
        curvature_reg=args.curvature_reg,
        ewc_lambda=args.ewc_lambda,
        ewc_gamma=args.ewc_gamma,
    )

    feature_store = FeatureStore(args.feature_storage_dir)
    splits: dict[str, dict[str, dict[str, object]]] = {}
    for dataset in datasets:
        retrieval = is_retrieval_dataset(dataset)
        logger.info("Loading dataset: {} (retrieval={})", dataset, retrieval)
        splits[dataset] = {
            "train": feature_store.load(dataset, "train", retrieval),
            "test": feature_store.load(dataset, "test", retrieval),
        }

    logger.info("Datasets loaded. Starting continual training over {} tasks...", len(datasets))

    first_task_name = datasets[0]
    first_task_test_features = splits[first_task_name]["test"]["features"].clone()

    score_matrix: list[list[float]] = []
    retrieval_matrices: dict[str, list[list[float]]] = {
        metric: [] for metric in RETRIEVAL_METRICS
    }
    model = None
    for task_index, dataset in enumerate(datasets):
        task_type = "retrieval" if is_retrieval_dataset(dataset) else "classification"
        logger.info(
            "=== Task {}/{}: {} ({}) ===",
            task_index + 1, len(datasets), dataset, task_type
        )
        model = trainer.fit_task(splits[dataset]["train"], model, task_type)

        row: list[float] = []
        retrieval_row = {metric: [] for metric in RETRIEVAL_METRICS}
        for seen_dataset in datasets[: task_index + 1]:
            if is_retrieval_dataset(seen_dataset):
                metrics = trainer.evaluate_retrieval(model, splits[seen_dataset]["test"])
                for metric in RETRIEVAL_METRICS:
                    retrieval_row[metric].append(metrics[metric])
                row.append(metrics["t2i_r5"])
                logger.info("{} retrieval: {}", seen_dataset, metrics)
            else:
                train = splits[seen_dataset]["train"]
                class_text = trainer.build_class_text(
                    train["text_features"].to(trainer.device),
                    train["labels"].long().to(trainer.device),
                )
                accuracy = trainer.evaluate_classification(
                    model, splits[seen_dataset]["test"], class_text
                )
                row.append(accuracy)
                logger.info("{} accuracy: {:.2f}%", seen_dataset, accuracy)

        score_matrix.append(row)
        for metric in RETRIEVAL_METRICS:
            retrieval_matrices[metric].append(retrieval_row[metric])

    drift_metrics = {}
    if model is not None and len(datasets) > 1:
        try:
            final_features = model.encode_image(
                first_task_test_features.to(trainer.device)
            ).cpu()
            drift_metrics = compute_representation_drift(
                first_task_test_features, final_features
            )
            logger.info("Representation drift (task-1 initial vs final): {}", drift_metrics)
            print("\n--- Representation Drift (Task 1 embeddings) ---")
            for k, v in drift_metrics.items():
                print(f"  {k:20s}: {v:.6f}")
        except Exception as exc:
            logger.warning("Drift computation skipped: {}", exc)

    results = _summarize(
        datasets,
        score_matrix,
        retrieval_matrices,
        REPORTED_RETRIEVAL_K[args.backbone],
    )
    results["method"] = args.method
    results["datasets"] = datasets
    results["drift"] = drift_metrics
    results["curvature_history"] = trainer.curvature_history
    results["config"] = {
        "backbone": args.backbone,
        "seed": args.seed,
        "method": args.method,
        "curvature": args.curv,
        "curvature_min": args.curvature_min,
        "curvature_max": args.curvature_max,
        "curvature_lr": args.curvature_lr,
        "curvature_reg": args.curvature_reg,
        "ewc_lambda": args.ewc_lambda,
        "ewc_gamma": args.ewc_gamma,
        "variance_ratio": variance_ratio,
        "pullback_beta": pullback_beta,
        "classification": vars(classification_cfg),
        "retrieval": vars(retrieval_cfg),
    }

    _print_benchmark_table(results, args.method)
    logger.info("Full results:\n{}", json.dumps(results, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        logger.info("Saved results to {}", args.output)


if __name__ == "__main__":
    main()
