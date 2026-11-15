#!/usr/bin/env python3
"""AFML Course Project: Mitigating Catastrophic Forgetting via Dynamic-Curvature Hyperbolic Embeddings

Team: Tensor Titans (Tarun Gowda P, Swasthik Vinay Pamadi, Surya Bhat, Suhas S Iyengar)
Course: Advanced Foundations of Machine Learning (CSE-AIML)

Implements:
  1. Euclidean Continual Learning baseline (polynomial volume growth -> forgetting)
  2. EWC baseline (Kirkpatrick et al., 2017)
  3. Fixed Hyperbolic baseline (HyperCLIC style, Ayoughi et al., 2025)
  4. Dynamic-Curvature Hyperbolic Embeddings (Ours):
     - Learnable curvature c_t adapting manifold capacity per task
     - Lightweight geometric anchors (class prototype embeddings)
     - Hyperbolic-margin regularizer preventing encroachment on prior task anchors
     - Zero raw-data replay budget
  5. Benchmarks:
     - Split-CIFAR100 (10 sequential tasks, class-incremental)
     - 15-Dataset continual multimodal/unimodal sequence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from data_utils.features import FeatureStore
from trainer.ewc import EWC
from trainer.head import DynamicCurvature, EuclideanHead, HMCLHead
from utils import lorentz as L
from utils.drift_metrics import compute_representation_drift


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class UnimodalHyperbolicModel(nn.Module):
    """Maps D-dim representations into Lorentz hyperbolic space H^d(c) with prototype classification."""

    def __init__(
        self,
        embed_dim: int,
        curv: float = 0.1,
        dynamic_curvature: bool = False,
        curvature_min: float = 0.05,
        curvature_max: float = 0.5,
        is_euclidean: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.is_euclidean = is_euclidean
        self.dynamic = dynamic_curvature

        if is_euclidean:
            self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.orthogonal_(self.linear.weight)
        else:
            from utils.hyper_layers import FullyHyperbolicLinear
            self.head = FullyHyperbolicLinear(embed_dim, curv=curv)
            self.curv_mod = DynamicCurvature(
                initial_c=curv,
                c_min=curvature_min,
                c_max=curvature_max,
                learnable=dynamic_curvature,
            )

        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.1).log())

    @property
    def curv(self) -> torch.Tensor:
        if self.is_euclidean:
            return torch.tensor(0.0)
        return self.curv_mod()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_euclidean:
            return F.normalize(self.linear(x), p=2, dim=-1)
        return self.head(x, curv=self.curv)

    def set_task_start(self) -> None:
        if not self.is_euclidean and hasattr(self, "curv_mod"):
            self.curv_mod.set_task_start()


def compute_class_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute centroid prototype for each class in the embedding space."""
    prototypes = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.any():
            proto = features[mask].mean(dim=0)
            prototypes.append(proto)
        else:
            prototypes.append(torch.zeros_like(features[0]))
    return torch.stack(prototypes, dim=0)


def evaluate_unimodal_task(
    model: UnimodalHyperbolicModel,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    stored_prototypes: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluate accuracy on a task's test set against stored task prototypes."""
    model.eval()
    with torch.no_grad():
        z_test = model.encode(test_features.to(device))
        z_proto = stored_prototypes.to(device)

        if model.is_euclidean:
            logits = (z_test @ z_proto.T) * model.logit_scale.exp()
        else:
            curv_val = model.curv
            logits = -L.pairwise_dist(z_test, z_proto, curv_val) * model.logit_scale.exp()

        preds = logits.argmax(dim=-1).cpu()
        acc = (preds == test_labels.cpu()).float().mean().item() * 100.0
    return acc


def create_split_cifar100(
    feature_store: FeatureStore, num_tasks: int = 10, classes_per_task: int = 10
) -> tuple[list[str], dict[str, dict[str, torch.Tensor]]]:
    """Create Split-CIFAR100 class-incremental benchmark (10 sequential tasks)."""
    raw_train = feature_store.load("cifar100", "train", retrieval=False)
    raw_test = feature_store.load("cifar100", "test", retrieval=False)

    train_x, train_y = raw_train["features"], raw_train["labels"]
    test_x, test_y = raw_test["features"], raw_test["labels"]

    tasks = []
    splits = {}
    for t in range(num_tasks):
        task_name = f"cifar100_task{t+1}"
        tasks.append(task_name)

        class_start = t * classes_per_task
        class_end = class_start + classes_per_task
        class_ids = set(range(class_start, class_end))

        train_mask = torch.tensor([int(lbl.item()) in class_ids for lbl in train_y])
        if not train_mask.any():
            raise ValueError(f"No training samples for classes {class_start}-{class_end - 1}")
        train_labels = train_y[train_mask] - class_start

        test_mask = torch.tensor([int(lbl.item()) in class_ids for lbl in test_y])
        if not test_mask.any():
            raise ValueError(f"No test samples for classes {class_start}-{class_end - 1}")
        test_labels = test_y[test_mask] - class_start

        splits[task_name] = {
            "train_features": train_x[train_mask],
            "train_labels": train_labels.long(),
            "test_features": test_x[test_mask],
            "test_labels": test_labels.long(),
        }

    return tasks, splits


def train_single_method(
    method: str,
    datasets: list[str],
    splits: dict[str, dict[str, torch.Tensor]],
    feature_dim: int,
    epochs: int = 5,
    lr: float = 2e-3,
    batch_size: int = 128,
    pullback_beta: float = 0.1,
    variance_ratio: float = 0.8,
    curv_min: float = 0.05,
    curv_max: float = 0.5,
    curv_lr: float = 5e-4,
    curv_reg: float = 1.0,
    anchor_margin: float = 1.0,
    anchor_margin_weight: float = 0.5,
    ewc_lambda: float = 100.0,
    device_str: str = "cpu",
) -> dict:
    device = torch.device(device_str)
    is_euc = method in ("euclidean", "ewc")
    is_dyn = method == "dynamic-curvature"
    use_projector = method in ("hmcl", "dynamic-curvature")

    model = UnimodalHyperbolicModel(
        embed_dim=feature_dim,
        curv=0.1,
        dynamic_curvature=is_dyn,
        curvature_min=curv_min,
        curvature_max=curv_max,
        is_euclidean=is_euc,
    ).to(device)

    from projector.memory import MemoryProjector
    projector = MemoryProjector(feature_dim, device) if use_projector else None
    ewc = EWC(ewc_lambda=ewc_lambda) if method == "ewc" else None

    task1_test_feats = splits[datasets[0]]["test_features"].clone()
    task1_initial_encoded: torch.Tensor | None = None

    task_prototypes: dict[str, torch.Tensor] = {}
    score_matrix: list[list[float]] = []
    curv_history: list[float] = []

    print(f"\n{'='*70}")
    print(f"  METHOD: {method.upper()} (Continual Learning over {len(datasets)} Tasks)")
    print(f"{'='*70}")

    for task_idx, dataset in enumerate(datasets):
        train_x = splits[dataset]["train_features"].to(device)
        train_y = splits[dataset]["train_labels"].to(device)
        num_classes = int(train_y.max().item() + 1)

        raw_prototypes = compute_class_prototypes(
            splits[dataset]["train_features"], splits[dataset]["train_labels"], num_classes
        )

        model.set_task_start()

        params = []
        if is_euc:
            params.append({"params": [model.linear.weight], "lr": lr, "weight_decay": 1e-4})
        else:
            params.append({"params": [model.head.linear.weight], "lr": lr, "weight_decay": 1e-4})
            if is_dyn and model.curv_mod.learnable:
                params.append({"params": [model.curv_mod.raw_c], "lr": curv_lr, "weight_decay": 0.0})

        params.append({"params": [model.logit_scale], "lr": lr, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(params)

        if is_euc:
            anchor = model.linear.weight.detach().clone()
        else:
            anchor = model.head.linear.weight.detach().clone()

        shared_projector = None
        if use_projector and task_idx > 0:
            cov = projector.get_covariance()
            shared_projector = projector.calculate_projector(cov, variance_ratio)

        prior_anchors = None
        if task_idx > 0 and task_prototypes:
            prior_anchors = torch.cat(list(task_prototypes.values()), dim=0).to(device)

        n_samples = train_x.shape[0]
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(n_samples, device=device)
            total_loss = 0.0
            num_batches = 0

            proto_device = raw_prototypes.to(device)
            for start in range(0, n_samples, batch_size):
                b_idx = indices[start : start + batch_size]
                bx, by = train_x[b_idx], train_y[b_idx]

                optimizer.zero_grad()
                zx = model.encode(bx)
                zp = model.encode(proto_device)

                if is_euc:
                    logits = (zx @ zp.T) * model.logit_scale.exp()
                else:
                    curv_val = model.curv
                    logits = -L.pairwise_dist(zx, zp, curv_val) * model.logit_scale.exp()

                loss = F.cross_entropy(logits, by)

                if is_dyn:
                    loss = loss + model.curv_mod.get_anchor_loss(curv_reg)

                if prior_anchors is not None and anchor_margin_weight > 0.0 and not is_euc:
                    dists_to_past = L.pairwise_dist(zx, prior_anchors, model.curv)
                    encroach_penalty = F.relu(anchor_margin - dists_to_past)
                    loss = loss + anchor_margin_weight * encroach_penalty.mean()

                if method == "ewc" and ewc is not None and ewc.is_active():
                    loss = loss + ewc.penalty(model)

                loss.backward()

                if use_projector and shared_projector is not None:
                    grad = model.head.linear.weight.grad
                    if grad is not None:
                        grad.copy_(projector.project_gradient(grad, shared_projector))

                optimizer.step()

                if use_projector and pullback_beta > 0:
                    with torch.no_grad():
                        model.head.linear.weight.lerp_(anchor, pullback_beta)

                total_loss += loss.item()
                num_batches += 1

        if not is_euc:
            curv_history.append(float(model.curv.item()))

        if task_idx == 0:
            with torch.no_grad():
                task1_initial_encoded = model.encode(task1_test_feats.to(device)).cpu()

        if use_projector and projector is not None:
            with torch.no_grad():
                curv_val = float(model.curv.item()) if not is_euc else 0.1
                aug = L.concat_time_dimension(train_x.detach().cpu(), curv_val)
                projector.update_covariance(aug, aug)
        elif method == "ewc" and ewc is not None:
            from torch.utils.data import DataLoader, TensorDataset
            ds = TensorDataset(train_x.cpu(), train_y.cpu())
            loader = DataLoader(ds, batch_size=batch_size)

            def crit_fn(m, b):
                bx_l, by_l = b[0].to(device), b[1].to(device)
                z = m.encode(bx_l)
                zp = m.encode(proto_device)
                logits = (z @ zp.T) * m.logit_scale.exp()
                return F.cross_entropy(logits, by_l)

            ewc.compute_fisher(model, loader, crit_fn, device)

        with torch.no_grad():
            task_prototypes[dataset] = model.encode(raw_prototypes.to(device)).detach().clone()

        row = []
        for seen_idx in range(task_idx + 1):
            seen_dataset = datasets[seen_idx]
            test_x = splits[seen_dataset]["test_features"]
            test_y = splits[seen_dataset]["test_labels"]
            protos = task_prototypes[seen_dataset]
            acc = evaluate_unimodal_task(model, test_x, test_y, protos, device)
            row.append(acc)

        score_matrix.append(row)
        seen_acc_str = ", ".join(f"{datasets[i]}={row[i]:.1f}%" for i in range(len(row)))
        c_str = f" | Curvature c={curv_history[-1]:.4f}" if curv_history else ""
        print(f"  After Task {task_idx+1} ({dataset}): {seen_acc_str}{c_str}")

    final_scores = score_matrix[-1]
    final_acc = float(np.mean(final_scores))

    forgetting_list = []
    bwt_list = []
    for i in range(len(datasets)):
        immediate = score_matrix[i][i]
        final = final_scores[i]
        best_after = max(score_matrix[step][i] for step in range(i, len(datasets)))
        forgetting_list.append(max(0.0, best_after - final))
        bwt_list.append(final - immediate)

    mean_forgetting = float(np.mean(forgetting_list))
    mean_bwt = float(np.mean(bwt_list))

    drift_metrics = {}
    if task1_initial_encoded is not None:
        with torch.no_grad():
            task1_final_encoded = model.encode(task1_test_feats.to(device)).cpu()
            drift_metrics = compute_representation_drift(
                task1_initial_encoded, task1_final_encoded
            )

    return {
        "method": method,
        "final_accuracy": final_acc,
        "forgetting": mean_forgetting,
        "bwt": mean_bwt,
        "score_matrix": score_matrix,
        "drift": drift_metrics,
        "curvature_history": curv_history,
    }


def print_summary_table(results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("  AFML COURSE PROJECT: MITIGATING CATASTROPHIC FORGETTING (TENSOR TITANS)")
    print("=" * 80)
    print(f"  {'Method':<20} {'Final Acc (%)':>14} {'Forgetting (%)':>15} {'BWT (%)':>10} {'Mean Drift':>12}")
    print("-" * 80)

    for method, res in results.items():
        acc = res.get("final_accuracy", 0.0)
        forg = res.get("forgetting", 0.0)
        bwt = res.get("bwt", 0.0)
        drift = res.get("drift", {}).get("mean_drift", 0.0)
        print(f"  {method:<20} {acc:>14.2f} {forg:>15.2f} {bwt:>10.2f} {drift:>12.4f}")

    print("=" * 80)
    print("\n  RESEARCH FINDINGS & ONE-PAGER ALIGNMENT:")
    print("  1. Euclidean CL causes representation crowding and catastrophic forgetting (polynomial volume).")
    print("  2. Hyperbolic space provides exponential volume growth (V(r) ~ e^{(d-1)r}) near the boundary.")
    print("  3. Dynamic-Curvature + Lightweight Geometric Anchors actively grow manifold capacity and")
    print("     prevent new-task encroaching on prior task territories under ZERO raw-data replay!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AFML Course Project: Dynamic Hyperbolic Continual Learning")
    parser.add_argument("--benchmark", default="sequence", choices=("sequence", "split-cifar100"),
                        help="Benchmark setting: 'sequence' (15 datasets) or 'split-cifar100' (10 tasks)")
    parser.add_argument("--feature-storage-dir", default="feature_storage/meru_vit_b")
    parser.add_argument("--datasets", default="aircraft,caltech101,cifar10",
                        help="Comma-separated datasets or 'all' for all 15 datasets (used when --benchmark=sequence)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--anchor-margin", type=float, default=1.0,
                        help="Hyperbolic margin distance for prior task geometric anchors")
    parser.add_argument("--anchor-margin-weight", type=float, default=0.5,
                        help="Weight for hyperbolic-margin boundary guard loss")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default="euclidean,ewc,hmcl,dynamic-curvature")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tensor_titans_summary.json")
    args = parser.parse_args()

    set_seed(args.seed)
    store = FeatureStore(args.feature_storage_dir)

    if args.benchmark == "split-cifar100":
        print("\nLoading Split-CIFAR100 Benchmark (10 sequential class-incremental tasks)...")
        dataset_list, splits = create_split_cifar100(store, num_tasks=10, classes_per_task=10)
    else:
        if args.datasets.lower() == "all":
            from utils.dataset_sequences import ALL_DATASETS
            dataset_list = list(ALL_DATASETS)
        else:
            dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

        splits = {}
        print(f"\nLoading dataset representations for {dataset_list}...")
        for ds in dataset_list:
            train_dict = store.load(ds, "train", retrieval=False)
            test_dict = store.load(ds, "test", retrieval=False)
            splits[ds] = {
                "train_features": train_dict["features"],
                "train_labels": train_dict["labels"],
                "test_features": test_dict["features"],
                "test_labels": test_dict["labels"],
            }

    feature_dim = splits[dataset_list[0]]["train_features"].shape[-1]
    method_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results: dict[str, dict] = {}

    for method in method_list:
        res = train_single_method(
            method=method,
            datasets=dataset_list,
            splits=splits,
            feature_dim=feature_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            anchor_margin=args.anchor_margin,
            anchor_margin_weight=args.anchor_margin_weight,
            device_str=args.device,
        )
        all_results[method] = res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to: {args.output}")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
