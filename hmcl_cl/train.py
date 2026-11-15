"""Continual-learning training loop, evaluation and metrics for all methods."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as L
from .geometry import compute_representation_drift
from .models import ContinualLearningModel
from .regularizers import EWC, MemoryProjector

ROOT = Path(__file__).resolve().parent.parent

METHOD_ALIASES = {
    "euclidean": "euclidean",
    "ewc": "ewc",
    "hmcl": "hmcl",
    "hyperbolic": "hmcl",
    "fixed": "hmcl",
    "dynamic": "dynamic-curvature",
    "dynamic-curvature": "dynamic-curvature",
    "ablation-euclidean": "ablation-euclidean",
    "ablation-fixed-curvature": "ablation-fixed-curvature",
}
METHOD_LABELS = {
    "euclidean": "Euclidean",
    "ewc": "Euclidean + EWC",
    "hmcl": "Hyperbolic (fixed c)",
    "dynamic-curvature": "Dynamic curvature (ours)",
    "ablation-euclidean": "Ablation: ours in Euclidean space",
    "ablation-fixed-curvature": "Ablation: ours with fixed curvature",
}
DYNAMIC_FAMILY = ("dynamic-curvature", "ablation-euclidean", "ablation-fixed-curvature")

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def encode_all(model, x, aug, bs=512) -> torch.Tensor:
    model.eval()
    return torch.cat([model.encode(aug(x[i:i + bs], train=False)) for i in range(0, len(x), bs)])


@torch.no_grad()
def compute_prototypes(model, x, y, num_classes, aug) -> torch.Tensor:
    """Class-mean prototypes. Euclidean: re-normalised to the sphere. Hyperbolic: mean of
    the spatial coordinates, i.e. exactly the prototype form used in the training loss."""
    z = encode_all(model, x, aug)
    protos = torch.stack([z[y == c].mean(0) for c in range(num_classes)])
    return F.normalize(protos, dim=-1) if model.is_euclidean else protos


@torch.no_grad()
def accuracy(model, x, y, protos, aug) -> float:
    z = encode_all(model, x, aug)
    return float((model.logits(z, protos).argmax(-1) == y).float().mean().item() * 100.0)


@torch.no_grad()
def compensate_prototype_drift(anchors: dict, z_old: torch.Tensor, z_new: torch.Tensor,
                               is_euclidean: bool) -> None:
    """Semantic drift compensation (Yu et al., CVPR 2020), no replay.

    Old-class prototypes cannot be recomputed (their data is gone), so each one is moved
    by the Gaussian-weighted mean displacement of current-task embeddings that started
    near it: p <- p + sum_i w_i (z_new_i - z_old_i) / sum_i w_i.
    """
    delta = z_new - z_old
    for name, protos in anchors.items():
        d2 = torch.cdist(protos, z_old).pow(2)
        sigma2 = d2.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        w = torch.exp(-d2 / (2 * sigma2))
        shift = (w @ delta) / w.sum(1, keepdim=True).clamp_min(1e-8)
        new = protos + shift
        anchors[name] = F.normalize(new, dim=-1) if is_euclidean else new


@torch.no_grad()
def class_radii(model, x, y, num_classes, protos, aug) -> torch.Tensor:
    """Per-class spread (RMS per-dimension std) around the prototype, for PASS-style augmentation."""
    z = encode_all(model, x, aug)
    return torch.stack([(z[y == c] - protos[c]).pow(2).mean().sqrt() for c in range(num_classes)])


def run_experiment(method: str, stream: dict, args, seed: int, device: torch.device, aug) -> dict:
    set_seed(seed)
    names = list(stream.keys())
    T = len(names)
    ours = method in DYNAMIC_FAMILY
    if ours:
        model = ContinualLearningModel(method, args.embed_dim, args.width,
                                       curv_init=0.1, curv_min=0.05, curv_max=1.0).to(device)
    else:
        model = ContinualLearningModel(method, args.embed_dim, args.width).to(device)
    use_amp = bool(args.amp) and device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
        model.use_amp = True
        model.encoder.to(memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    use_projector = method == "hmcl"
    projector = MemoryProjector(args.embed_dim, device) if use_projector else None
    pullback_beta = 0.0 if ours else args.pullback_beta
    margin_weight = 0.0 if ours else args.anchor_margin_weight
    curv_lr = args.dynamic_curv_lr if ours else args.curv_lr
    curv_anchor_weight = args.dynamic_curv_anchor_weight if ours else args.curv_anchor_weight
    if method == "ewc":
        ewc = EWC(ewc_lambda=args.ewc_lambda)
    elif ours:
        ewc = EWC(ewc_lambda=args.dynamic_ewc_lambda)
    else:
        ewc = None

    anchors: dict[str, torch.Tensor] = {}
    radii: dict[str, torch.Tensor] = {}
    til, cil, curv_hist = [], [], []
    task1_z0 = None
    t0 = time.time()
    print(f"\n{'=' * 72}\n  {METHOD_LABELS[method]} [{method}]  seed={seed}  ({T} tasks)\n{'=' * 72}")

    for ti, tn in enumerate(names):
        d = stream[tn]
        tx, ty, K = d["train_x"], d["train_y"], d["num_classes"]
        model.set_task_start()

        teacher = None
        if ours and ti > 0:
            import copy
            teacher = copy.deepcopy(model).eval()
            for p_ in teacher.parameters():
                p_.requires_grad_(False)

        groups = [
            {"params": model.encoder.parameters(), "lr": args.lr, "weight_decay": 1e-4},
            {"params": model.head.parameters(), "lr": args.lr, "weight_decay": 1e-4},
            {"params": [model.logit_scale], "lr": args.lr, "weight_decay": 0.0},
        ]
        if model.is_dynamic:
            groups.append({"params": [model.curv_mod.raw_c], "lr": curv_lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
        steps = args.epochs * ((len(tx) + args.batch_size - 1) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.15)

        head_anchor = model.head_weight().detach().clone()
        P = None
        if use_projector and ti > 0:
            P = projector.calculate_projector(projector.get_covariance(), args.variance_ratio)
        prior = torch.cat(list(anchors.values())) if anchors else None
        prior_r = torch.cat(list(radii.values())) if radii else None
        joint = ours and prior is not None
        model.train()
        for ep in range(args.epochs):
            perm = torch.randperm(len(tx), device=device)
            tot, nb = 0.0, 0
            for s in range(0, len(tx), args.batch_size):
                idx = perm[s:s + args.batch_size]
                bx, by = aug(tx[idx], train=True), ty[idx]
                opt.zero_grad(set_to_none=True)
                z = model.encode(bx)

                present = torch.stack([(by == c).any() for c in range(K)])
                protos = torch.stack([z[by == c].mean(0) if present[c] else z.new_zeros(z.shape[1])
                                      for c in range(K)])
                if joint:
                    n_old = prior.shape[0]
                    all_p = torch.cat([prior, protos])
                    mask = torch.cat([present.new_ones(n_old), present]).view(1, -1)
                    lg = model.logits(z, all_p).masked_fill(~mask, -1e4)
                    loss = F.cross_entropy(lg, by + n_old)
                    if args.dynamic_proto_aug_weight > 0:
                        j = torch.randint(0, n_old, (len(by),), device=z.device)
                        fake = prior[j] + torch.randn_like(prior[j]) * prior_r[j].unsqueeze(1)
                        lg_f = model.logits(fake, all_p).masked_fill(~mask, -1e4)
                        loss = loss + args.dynamic_proto_aug_weight * F.cross_entropy(lg_f, j)
                else:
                    lg = model.logits(z, protos).masked_fill(~present.view(1, -1), -1e4)
                    loss = F.cross_entropy(lg, by)

                if model.is_dynamic:
                    loss = loss + model.curv_mod.get_anchor_loss(curv_anchor_weight)
                if prior is not None and not model.is_euclidean and margin_weight > 0:
                    dist = L.pairwise_dist(z, prior, model.curv)
                    loss = loss + margin_weight * F.relu(args.anchor_margin - dist).mean()
                if ewc is not None and ewc.is_active():
                    loss = loss + ewc.penalty(model)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if P is not None and model.head_weight().grad is not None:
                    g = model.head_weight().grad
                    g.copy_(projector.project_gradient(g, P))
                scaler.step(opt)
                scaler.update()
                sched.step()
                if use_projector and pullback_beta > 0:
                    with torch.no_grad():
                        model.head_weight().lerp_(head_anchor, pullback_beta)
                tot += float(loss.item())
                nb += 1
            if args.verbose or ep == args.epochs - 1:
                print(f"    task {ti+1} epoch {ep+1}/{args.epochs} loss={tot / nb:.4f}")

        if not model.is_euclidean:
            curv_hist.append(float(model.curv.item()))
        if ti == 0:
            task1_z0 = encode_all(model, stream[names[0]]["test_x"], aug).clone()

        if teacher is not None:
            compensate_prototype_drift(anchors, encode_all(teacher, tx, aug), encode_all(model, tx, aug),
                                       model.is_euclidean)
        anchors[tn] = compute_prototypes(model, tx, ty, K, aug)
        if ours:
            radii[tn] = class_radii(model, tx, ty, K, anchors[tn], aug)

        if projector is not None:
            with torch.no_grad():
                h = encode_all(model, tx, aug)
                aug_h = L.concat_time_dimension(h, float(model.curv.item()))
                projector.update_covariance(aug_h, aug_h)
        if ewc is not None:
            from torch.utils.data import DataLoader, TensorDataset
            loader = DataLoader(TensorDataset(torch.arange(len(tx))), batch_size=args.batch_size, shuffle=True)
            p_t = anchors[tn]

            def crit(m, b):
                i = b[0].to(device)
                return F.cross_entropy(m.logits(m.encode(aug(tx[i], train=False)), p_t), ty[i])

            ewc.compute_fisher(model, loader, crit, device)
            model.train()

        model.eval()
        row_til, row_cil = [], []
        all_protos = torch.cat([anchors[names[j]] for j in range(ti + 1)])
        for j in range(ti + 1):
            dj = stream[names[j]]
            row_til.append(accuracy(model, dj["test_x"], dj["test_y"], anchors[names[j]], aug))
            row_cil.append(accuracy(model, dj["test_x"], dj["test_y"] + dj["class_offset"], all_protos, aug))
        til.append(row_til)
        cil.append(row_cil)
        cs = f" | c={curv_hist[-1]:.4f}" if curv_hist else ""
        print(f"  After task {ti+1}: task-IL avg={np.mean(row_til):.2f}%  class-IL avg={np.mean(row_cil):.2f}%"
              f"  [task-IL per task: {', '.join(f'{a:.1f}' for a in row_til)}]{cs}")
        model.train()

    def summarise(mat):
        final = mat[-1]
        forg = [max(mat[s][i] for s in range(i, T)) - final[i] for i in range(T - 1)]
        bwt = [final[i] - mat[i][i] for i in range(T - 1)]
        return {
            "final_accuracy": float(np.mean(final)),
            "forgetting": float(np.mean(forg)) if forg else 0.0,
            "bwt": float(np.mean(bwt)) if bwt else 0.0,
            "learning_accuracy": float(np.mean([mat[i][i] for i in range(T)])),
            "score_matrix": mat,
        }

    z1 = encode_all(model, stream[names[0]]["test_x"], aug)
    drift = compute_representation_drift(
        z1, task1_z0, curv=model.curv.to(device) if not model.is_euclidean else 0.1,
        is_hyperbolic=not model.is_euclidean)
    res = {
        "method": method, "seed": seed,
        "task_il": summarise(til), "class_il": summarise(cil),
        "drift_task1": drift, "curvature_history": curv_hist,
        "train_minutes": (time.time() - t0) / 60.0,
        "epochs": args.epochs,
        "amp": use_amp,
    }
    print(f"  -> task-IL acc {res['task_il']['final_accuracy']:.2f}% forgetting {res['task_il']['forgetting']:.2f}% | "
          f"class-IL acc {res['class_il']['final_accuracy']:.2f}% forgetting {res['class_il']['forgetting']:.2f}% "
          f"({res['train_minutes']:.1f} min)")
    return res


def print_table(results: dict[str, list[dict]]) -> None:
    def ms(vals):
        return f"{np.mean(vals):6.2f} ± {np.std(vals):5.2f}" if len(vals) > 1 else f"{vals[0]:6.2f}"

    for proto in ("task_il", "class_il"):
        title = "TASK-INCREMENTAL (task id given)" if proto == "task_il" else "CLASS-INCREMENTAL (no task id)"
        print("\n" + "=" * 86)
        print(f"  {title}   mean ± std over {len(next(iter(results.values())))} seed(s)")
        print("=" * 86)
        print(f"  {'Method':<26}{'Final Acc (%)':>18}{'Forgetting (%)':>20}{'BWT (%)':>20}")
        print("-" * 86)
        for m, runs in results.items():
            print(f"  {METHOD_LABELS[m]:<26}{ms([r[proto]['final_accuracy'] for r in runs]):>18}"
                  f"{ms([r[proto]['forgetting'] for r in runs]):>20}{ms([r[proto]['bwt'] for r in runs]):>20}")
        best = min(results, key=lambda m: np.mean([r[proto]["forgetting"] for r in results[m]]))
        print(f"  Lowest forgetting: {METHOD_LABELS[best]}")
    print("=" * 86)

    if "ewc" in results and len(results["ewc"]) > 1:
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            ttest_rel = None
        print("\n  Forgetting vs Euclidean-EWC (paired over seeds, class-IL):")
        base = [r["class_il"]["forgetting"] for r in results["ewc"]]
        for m, runs in results.items():
            if m == "ewc":
                continue
            v = [r["class_il"]["forgetting"] for r in runs]
            diff = np.mean(v) - np.mean(base)
            p = ttest_rel(v, base).pvalue if ttest_rel else float("nan")
            print(f"    {METHOD_LABELS[m]:<26} Δ forgetting = {diff:+.2f} pts   p = {p:.4f}")


def make_args(**overrides) -> argparse.Namespace:
    """Default hyper-parameters as a Namespace (for notebook use), with overrides."""
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise KeyError(f"unknown option {k}")
        setattr(args, k, v)
    return args


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Real-image continual learning: 4 methods")
    ap.add_argument("--dataset", default="split_cifar100", choices=["split_cifar100", "split_miniimagenet"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--source", default="kaggle", choices=["kaggle", "torchvision-hf"],
                    help="kaggle (default, via kagglehub) or torchvision (CIFAR) / HuggingFace (MiniImageNet)")
    ap.add_argument("--kaggle-dataset", default=None, help="override Kaggle slug, e.g. owner/name")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--classes-per-task", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=64, help="MiniImageNet resize (CIFAR is 32)")
    ap.add_argument("--methods", nargs="+",
                    default=["euclidean", "ewc", "hmcl", "dynamic-curvature"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--width", type=int, default=64, help="encoder base channel width")
    ap.add_argument("--curv-lr", type=float, default=1e-3)
    ap.add_argument("--curv-anchor-weight", type=float, default=1.0)
    ap.add_argument("--anchor-margin", type=float, default=1.2)
    ap.add_argument("--anchor-margin-weight", type=float, default=1.0)
    ap.add_argument("--pullback-beta", type=float, default=0.15)
    ap.add_argument("--variance-ratio", type=float, default=0.85)
    ap.add_argument("--ewc-lambda", type=float, default=1000.0)
    ap.add_argument("--dynamic-ewc-lambda", type=float, default=5000.0, help="Fisher (EWC) strength in our method")
    ap.add_argument("--dynamic-proto-aug-weight", type=float, default=1.0, help="PASS-style prototype augmentation")
    ap.add_argument("--dynamic-curv-lr", type=float, default=1e-2, help="learning rate of the curvature")
    ap.add_argument("--dynamic-curv-anchor-weight", type=float, default=0.1, help="pull of c towards its task-start value")
    ap.add_argument("--max-train-per-task", type=int, default=0, help="subsample for quick tests (0 = all)")
    ap.add_argument("--max-test-per-task", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--amp", type=int, default=0, help="1 = fp16 mixed-precision encoder (faster on GPU)")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "real_benchmark.json")
    return ap
