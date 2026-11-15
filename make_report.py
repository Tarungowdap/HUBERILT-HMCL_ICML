#!/usr/bin/env python3
"""Turn saved result JSONs into a GitHub-ready report: tables, improvement %, charts.

    python make_report.py --results /content/drive/MyDrive/hmcl_results_gpu --out results --readme README.md

Writes:
    <out>/RESULTS.md                       one page with every table and chart (renders on GitHub)
    <out>/<dataset>/results_table.md       mean ± std per method
    <out>/<dataset>/improvement_table.md   ours vs each baseline: % less forgetting, accuracy gain, p-value
    <out>/<dataset>/*.png                  charts
With --readme README.md the tables and charts are also written into the README, between the
<!-- RESULTS:START --> and <!-- RESULTS:END --> markers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OURS = "dynamic-curvature"
LABEL = {
    "euclidean": "Euclidean",
    "ewc": "Euclidean + EWC",
    "hmcl": "Hyperbolic (fixed c)",
    "dynamic-curvature": "Dynamic curvature (ours)",
    "ablation-euclidean": "Ablation: Euclidean space",
    "ablation-fixed-curvature": "Ablation: fixed curvature",
}
COLOR = {
    "euclidean": "#2a78d6", "ewc": "#eb6834", "hmcl": "#1baf7a", "dynamic-curvature": "#e87ba4",
    "ablation-euclidean": "#008300", "ablation-fixed-curvature": "#6250d6",
}
PROTOS = (("task_il", "Task-IL"), ("class_il", "Class-IL"))
DS_NAME = {"split_cifar100": "Split-CIFAR100", "split_miniimagenet": "Split-MiniImageNet"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(results_dir: str) -> dict:
    """runs[dataset][method][seed] = result dict (layout: <results>/<dataset>/<method>__seed<k>.json)."""
    runs = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(results_dir, "*", "*__seed*.json")):
        r = json.load(open(f))
        ds = r.get("dataset") or os.path.basename(os.path.dirname(f))
        runs[ds][r["method"]][r["seed"]] = r
    return runs


def ms(v) -> str:
    return f"{np.mean(v):.2f} ± {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(colors=MUTED)


def results_table(runs, order) -> str:
    head = "| Method | Seeds | Task-IL acc | Task-IL forgetting | Class-IL acc | Class-IL forgetting | Curvature c |\n"
    head += "|---|---|---|---|---|---|---|\n"
    rows = []
    for m in order:
        rs = list(runs[m].values())
        c = rs[0].get("curvature_history") or []
        cells = [LABEL[m] if m != OURS else f"**{LABEL[m]}**", str(len(rs))]
        for p, _ in PROTOS:
            cells += [ms([r[p]["final_accuracy"] for r in rs]), ms([r[p]["forgetting"] for r in rs])]
        cells.append(f"{c[0]:.3f} → {c[-1]:.3f}" if c else "—")
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def improvement(runs, ref):
    """Ours vs one baseline over paired seeds."""
    seeds = sorted(set(runs[OURS]) & set(runs[ref]))
    out = {}
    for p, _ in PROTOS:
        a_f = np.array([runs[OURS][s][p]["forgetting"] for s in seeds])
        b_f = np.array([runs[ref][s][p]["forgetting"] for s in seeds])
        a_a = np.array([runs[OURS][s][p]["final_accuracy"] for s in seeds])
        b_a = np.array([runs[ref][s][p]["final_accuracy"] for s in seeds])
        pval = None
        if len(seeds) > 1:
            from scipy.stats import ttest_rel
            pval = float(ttest_rel(a_f, b_f).pvalue)
            pval = None if np.isnan(pval) else pval
        out[p] = dict(
            forg_ref=b_f.mean(), forg_ours=a_f.mean(),
            forg_reduction_pct=100 * (b_f.mean() - a_f.mean()) / max(b_f.mean(), 1e-6),
            acc_ref=b_a.mean(), acc_ours=a_a.mean(), acc_gain_pts=a_a.mean() - b_a.mean(),
            acc_gain_pct=100 * (a_a.mean() - b_a.mean()) / max(b_a.mean(), 1e-6),
            p=pval, seeds=len(seeds))
    return out


def improvement_table(imps) -> str:
    head = ("| Compared with | Setting | Forgetting (baseline → ours) | Forgetting reduction | "
            "Accuracy (baseline → ours) | Accuracy gain | p-value (forgetting) |\n|---|---|---|---|---|---|---|\n")
    rows = []
    for ref, imp in imps.items():
        for p, tag in PROTOS:
            d = imp[p]
            red = f"**{d['forg_reduction_pct']:.0f}% less**" if d["forg_reduction_pct"] >= 0 \
                else f"{-d['forg_reduction_pct']:.0f}% more"
            gain = f"{d['acc_gain_pts']:+.2f} pts ({d['acc_gain_pct']:+.0f}%)"
            pv = f"{d['p']:.4f}" if d["p"] is not None else ("n/a (1 seed)" if d["seeds"] < 2 else "n/a")
            rows.append(f"| {LABEL[ref]} | {tag} | {d['forg_ref']:.2f} → {d['forg_ours']:.2f} | {red} | "
                        f"{d['acc_ref']:.2f} → {d['acc_ours']:.2f} | {gain} | {pv} |")
    return head + "\n".join(rows) + "\n"


def chart_bars(runs, order, ds, path):
    """Forgetting and final accuracy per method (task-IL solid, class-IL hatched), mean ± std."""
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "forgetting", "Forgetting (lower is better)"),
                           (axes[1], "final_accuracy", "Final accuracy (higher is better)")):
        ax.set_facecolor(SURFACE)
        for (p, tag), off, hatch in zip(PROTOS, (-0.2, 0.2), (None, "//")):
            vals = [np.mean([r[p][key] for r in runs[m].values()]) for m in order]
            errs = [np.std([r[p][key] for r in runs[m].values()]) for m in order]
            bars = ax.bar(x + off, vals, 0.38, yerr=errs, capsize=3, color=[COLOR[m] for m in order],
                          hatch=hatch, edgecolor=SURFACE, linewidth=2, label=f"{tag} ({'hatched' if hatch else 'solid'})")
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, color=MUTED)
        ax.set_xticks(x, [LABEL[m] for m in order], rotation=10, fontsize=9)
        style(ax, f"{DS_NAME.get(ds, ds)}: {title}", "%")
        ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, facecolor=SURFACE); plt.close()


def chart_curves(runs, order, ds, path):
    """Average accuracy on all tasks seen so far, after each task (mean over seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), facecolor=SURFACE)
    for ax, (p, tag) in zip(axes, PROTOS):
        ax.set_facecolor(SURFACE)
        for m in order:
            curves = np.array([[np.mean(row) for row in r[p]["score_matrix"]] for r in runs[m].values()])
            mean = curves.mean(0); t = np.arange(1, len(mean) + 1)
            ax.plot(t, mean, lw=2, marker="o", ms=8, color=COLOR[m], label=LABEL[m],
                    markeredgecolor=SURFACE, markeredgewidth=2)
            if len(curves) > 1:
                ax.fill_between(t, mean - curves.std(0), mean + curves.std(0), color=COLOR[m], alpha=0.12, lw=0)
            ax.annotate(f"{mean[-1]:.1f}", (t[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED)
        ax.set_xticks(t)
        ax.set_xlabel("Tasks learned", color=MUTED)
        style(ax, f"{DS_NAME.get(ds, ds)}: {tag} accuracy on all seen tasks", "Accuracy (%)")
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150, facecolor=SURFACE); plt.close()


def chart_task_forgetting(runs, order, ds, path):
    """Histogram-style bars: forgetting of every old task at the end (task-IL, mean over seeds)."""
    T = len(next(iter(runs[order[0]].values()))["task_il"]["score_matrix"])
    width = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=SURFACE); ax.set_facecolor(SURFACE)
    for k, m in enumerate(order):
        per_task = []
        for r in runs[m].values():
            mat = r["task_il"]["score_matrix"]
            per_task.append([max(mat[s][i] for s in range(i, T)) - mat[-1][i] for i in range(T - 1)])
        ax.bar(np.arange(1, T) + (k - (len(order) - 1) / 2) * width, np.mean(per_task, 0), width,
               color=COLOR[m], edgecolor=SURFACE, linewidth=1, label=LABEL[m])
    ax.set_xticks(np.arange(1, T), [f"Task {i}" for i in range(1, T)])
    style(ax, f"{DS_NAME.get(ds, ds)}: forgetting of each old task after the last task (task-IL)", "Forgetting (%)")
    ax.legend(frameon=False, fontsize=8, ncol=min(len(order), 3), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(); plt.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close()


def chart_improvement(imps, ds, path):
    """% less forgetting of ours vs each baseline (task-IL and class-IL)."""
    refs = list(imps)
    x = np.arange(len(refs))
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE); ax.set_facecolor(SURFACE)
    for (p, tag), off, hatch in zip(PROTOS, (-0.2, 0.2), (None, "//")):
        vals = [imps[r][p]["forg_reduction_pct"] for r in refs]
        bars = ax.bar(x + off, vals, 0.38, color=[COLOR[r] for r in refs], hatch=hatch,
                      edgecolor=SURFACE, linewidth=2, label=f"{tag} ({'hatched' if hatch else 'solid'})")
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, [f"vs {LABEL[r]}" for r in refs], fontsize=9)
    style(ax, f"{DS_NAME.get(ds, ds)}: how much less our method forgets", "Forgetting reduction (%)")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, facecolor=SURFACE); plt.close()


README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def update_readme(readme: str, sections: list[str], out: str, note: str) -> None:
    """Replace the block between the RESULTS markers in README with the generated results."""
    text = open(readme).read()
    if README_START not in text or README_END not in text:
        raise SystemExit(f"{readme} has no {README_START} ... {README_END} block")
    rel = os.path.relpath(out, os.path.dirname(os.path.abspath(readme)) or ".")
    body = [README_START, "", "## Results (GPU, full Split benchmarks)", "",
            "*Generated by `make_report.py` from the saved result files. Do not edit by hand.*", ""]
    if note:
        body += [f"> {note}", ""]
    body += [s.replace("](RESULTS_DIR/", f"]({rel}/") for s in sections]
    body += ["", f"Full report with every chart: [`{rel}/RESULTS.md`]({rel}/RESULTS.md)", "", README_END]
    a, b = text.index(README_START), text.index(README_END) + len(README_END)
    open(readme, "w").write(text[:a] + "\n".join(body) + text[b:])
    print(f"README updated: {readme}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with <dataset>/<method>__seed<k>.json")
    ap.add_argument("--out", default="results", help="output folder (commit this to GitHub)")
    ap.add_argument("--readme", default=None, help="also write the results into this README (RESULTS markers)")
    ap.add_argument("--note", default="", help="one-line description of the run (shown at the top)")
    args = ap.parse_args()

    runs = load(args.results)
    if not runs:
        raise SystemExit(f"No result files found under {args.results}/<dataset>/")
    md = ["# Results\n"]
    if args.note:
        md.append(f"> {args.note}\n")
    md.append("Class-incremental continual learning, 10 tasks × 10 classes, zero replay. Values are mean ± std "
              "over seeds. Forgetting = best − final accuracy on old tasks (lower is better).\n")
    readme_sections = []
    for ds in sorted(runs):
        order = [m for m in LABEL if m in runs[ds]]
        d = os.path.join(args.out, ds); os.makedirs(d, exist_ok=True)
        rs0 = next(iter(runs[ds][order[0]].values()))
        seeds = sorted({s for m in order for s in runs[ds][m]})
        head = f"\n## {DS_NAME.get(ds, ds)}\n"
        info = f"Seeds: {seeds} · epochs: {rs0.get('epochs')}\n"
        md += [head, info]
        sec = [f"### {DS_NAME.get(ds, ds)}", "", info]

        t = results_table(runs[ds], order)
        open(os.path.join(d, "results_table.md"), "w").write(t)
        md += ["\n### Results\n", t]
        sec += ["**Results** (mean ± std over seeds)", "", t]

        refs = [m for m in ("euclidean", "ewc", "hmcl", "ablation-euclidean", "ablation-fixed-curvature") if m in runs[ds]]
        if OURS in runs[ds] and refs:
            imps = {r: improvement(runs[ds], r) for r in refs}
            it = improvement_table(imps)
            open(os.path.join(d, "improvement_table.md"), "w").write(it)
            json.dump(imps, open(os.path.join(d, "improvement.json"), "w"), indent=1)
            md += ["\n### How much better is our method?\n", it]
            sec += ["**How much better is Dynamic curvature (ours)?**", "", it]
            chart_improvement(imps, ds, os.path.join(d, "improvement.png"))
            md.append(f"\n![Forgetting reduction]({ds}/improvement.png)\n")

        chart_bars(runs[ds], order, ds, os.path.join(d, "forgetting_accuracy.png"))
        chart_curves(runs[ds], order, ds, os.path.join(d, "accuracy_over_tasks.png"))
        chart_task_forgetting(runs[ds], order, ds, os.path.join(d, "forgetting_per_task.png"))
        md += ["\n### Charts\n",
               f"![Forgetting and accuracy]({ds}/forgetting_accuracy.png)\n",
               f"![Accuracy over tasks]({ds}/accuracy_over_tasks.png)\n",
               f"![Forgetting per task]({ds}/forgetting_per_task.png)\n"]
        for name, alt in (("improvement", "Forgetting reduction"), ("forgetting_accuracy", "Forgetting and accuracy"),
                          ("accuracy_over_tasks", "Accuracy over tasks"), ("forgetting_per_task", "Forgetting per task")):
            if os.path.exists(os.path.join(d, f"{name}.png")):
                sec += [f"![{alt}](RESULTS_DIR/{ds}/{name}.png)", ""]
        readme_sections.append("\n".join(sec))

    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "RESULTS.md"), "w").write("\n".join(md))
    print(f"Report written to {args.out}/RESULTS.md")
    if args.readme:
        update_readme(args.readme, readme_sections, args.out, args.note)


if __name__ == "__main__":
    main()
