"""
Analyze entropy-probe CSVs produced by scripts/probe_entropy.py.

Level convention: level 0 = coarse sketch, levels 1..K-1 = refinements.

Usage:
    python scripts/analyze.py --csv results/sudoku-hard/entropy_probe_100.csv --tag hard
    python scripts/analyze.py --csv results/sudoku-easy/entropy_probe_100.csv --csv results/sudoku-medium/entropy_probe_100.csv --csv results/sudoku-hard/entropy_probe_100.csv --tag alldiff
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path, dataset_tag):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "dataset": dataset_tag,
                "sample_id": int(r["sample_id"]),
                "level": int(r["level"]),
                "L_cur": int(r["L_cur"]),
                "V_k": int(r["V_k"]),
                "H_mean": float(r["H_mean"]),
                "H_max": float(r["H_max"]),
                "C_mean": float(r["C_mean"]),
                "C_min": float(r["C_min"]),
                "correct": int(r["correct"]),
            })
    return rows


def group_by_level(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["level"])].append(r)
    return dict(sorted(g.items()))


def group_by_sample(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["sample_id"])].append(r)
    for k in g:
        g[k].sort(key=lambda x: x["level"])
    return dict(sorted(g.items()))


def sample_correctness(rows):
    out = {}
    for r in rows:
        out[(r["dataset"], r["sample_id"])] = r["correct"]
    return out


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan"), len(x)


def auc_score(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    tie_rank = np.zeros(len(counts))
    for i, c in enumerate(counts):
        tie_rank[i] = ranks[inv == i].mean()
    ranks = tie_rank[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def fig_entropy_vs_level(by_level, tag, outdir):
    # Group by dataset for plotting
    datasets = sorted(set(k[0] for k in by_level))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    for ds in datasets:
        levels = sorted(l for (d, l) in by_level if d == ds)
        means = [np.mean([r["H_mean"] for r in by_level[(ds, l)]]) for l in levels]
        q25 = [np.percentile([r["H_mean"] for r in by_level[(ds, l)]], 25) for l in levels]
        q75 = [np.percentile([r["H_mean"] for r in by_level[(ds, l)]], 75) for l in levels]
        ax1.plot(levels, means, marker="o", label=f"{ds} mean H")
        ax1.fill_between(levels, q25, q75, alpha=0.15)
    ax1.set_xlabel("hierarchy level (0 = coarse sketch)")
    ax1.set_ylabel("mean entropy (nats)")
    ax1.set_title(f"{tag}: entropy across hierarchy levels")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_entropy_vs_level.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_per_position(by_level, tag, outdir):
    datasets = sorted(set(k[0] for k in by_level))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    for ds in datasets:
        levels = sorted(l for (d, l) in by_level if d == ds)
        h_max = [np.mean([r["H_max"] for r in by_level[(ds, l)]]) for l in levels]
        ax1.plot(levels, h_max, marker="o", label=f"{ds} mean H_max")
    ax1.set_xlabel("hierarchy level (0 = coarse sketch)")
    ax1.set_ylabel("H_max (nats)")
    ax1.set_title(f"{tag}: worst-position uncertainty")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_per_position.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_entropy_vs_correctness(by_sample, tag, outdir):
    datasets = sorted(set(k[0] for k in by_sample))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    for i, ds in enumerate(datasets):
        correct = [k for k in by_sample if k[0] == ds and all(r["correct"] == 1 for r in by_sample[k])]
        wrong = [k for k in by_sample if k[0] == ds and all(r["correct"] == 0 for r in by_sample[k])]
        ax = axes[0][i]
        if not wrong:
            ax.text(0.5, 0.5, f"No failures in {ds}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            h0_c = [by_sample[k][0]["H_mean"] for k in correct]
            h0_w = [by_sample[k][0]["H_mean"] for k in wrong]
            ax.boxplot([h0_c, h0_w], tick_labels=[f"correct (n={len(h0_c)})", f"incorrect (n={len(h0_w)})"], widths=0.5)
            ax.set_title(f"{ds}: coarse entropy vs. correctness")
            ax.set_ylabel("H_mean at level 0")
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_entropy_vs_correctness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_predictive_auc(by_level, by_sample, tag, outdir):
    datasets = sorted(set(k[0] for k in by_level))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ds in datasets:
        labels = {k: v for k, v in sample_correctness([r for k, v in by_sample.items() if k[0] == ds for r in v]).items() if k[0] == ds}
        levels = sorted(l for (d, l) in by_level if d == ds)
        aucs = []
        for l in levels:
            ids = [(r["dataset"], r["sample_id"]) for r in by_level[(ds, l)]]
            scores = [-r["H_mean"] for r in by_level[(ds, l)]]
            labs = [labels[i] for i in ids]
            aucs.append(auc_score(scores, labs))
        ax.plot(levels, aucs, marker="o", label=f"{ds} AUC")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.6, label="chance")
    ax.set_xlabel("hierarchy level (0 = coarse sketch)")
    ax.set_ylabel("AUC of -H_mean predicting correctness")
    ax.set_title(f"{tag}: which level's entropy best predicts correctness?")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_predictive_auc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary_text(rows, by_level, by_sample, tag):
    datasets = sorted(set(k[0] for k in by_level))
    lines = [f"=== {tag} ==="]
    for ds in datasets:
        ds_labels = {k: v for k, v in sample_correctness([r for k, v in by_sample.items() if k[0] == ds for r in v]).items() if k[0] == ds}
        n = len(ds_labels)
        n_correct = sum(ds_labels.values())
        n_wrong = n - n_correct
        lines.append(f"\n--- {ds} (n={n}, correct={n_correct}, incorrect={n_wrong}) ---")
        lines.append("level | L_cur | V_k | H_mean | H_max | C_mean | C_min")
        for l in sorted(l for (d, l) in by_level if d == ds):
            rs = by_level[(ds, l)]
            lines.append(
                f"{l:5d} | {rs[0]['L_cur']:5d} | {rs[0]['V_k']:3d} | "
                f"{np.mean([r['H_mean'] for r in rs]):6.4f} | "
                f"{np.mean([r['H_max'] for r in rs]):6.4f} | "
                f"{np.mean([r['C_mean'] for r in rs]):6.4f} | "
                f"{np.mean([r['C_min'] for r in rs]):6.4f}"
            )

        lines.append("\nCorrelation with correctness (coarse level 0):")
        for col in ("H_mean", "H_max", "C_min"):
            vals = [by_sample[k][0][col] for k in by_sample if k[0] == ds]
            y = [ds_labels[k] for k in by_sample if k[0] == ds]
            if n_wrong > 1:
                rho, _ = spearman(vals, y)
                lines.append(f"  level-0 {col:6s}: spearman rho = {rho:+.4f}")
            else:
                lines.append(f"  level-0 {col:6s}: N/A (insufficient failures)")

        lines.append("\nAUC by level:")
        for l in sorted(l for (d, l) in by_level if d == ds):
            ids = [(r["dataset"], r["sample_id"]) for r in by_level[(ds, l)]]
            scores = [-r["H_mean"] for r in by_level[(ds, l)]]
            labs = [ds_labels[i] for i in ids]
            auc = auc_score(scores, labs)
            lines.append(f"  level {l:2d}: AUC = {auc:.4f}" if not np.isnan(auc) else f"  level {l:2d}: AUC = N/A")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    all_rows = []
    for path in a.csv:
        if not os.path.exists(path):
            print(f"[analyze] missing: {path}")
            continue
        # derive dataset tag from folder name
        ds_tag = os.path.basename(os.path.dirname(path))
        rows = load_csv(path, ds_tag)
        all_rows.extend(rows)

    if not all_rows:
        print("[analyze] no rows loaded, exiting.")
        return

    by_level = group_by_level(all_rows)
    by_sample = group_by_sample(all_rows)

    p1 = fig_entropy_vs_level(by_level, a.tag, a.outdir)
    p2 = fig_per_position(by_level, a.tag, a.outdir)
    p3 = fig_entropy_vs_correctness(by_sample, a.tag, a.outdir)
    p4 = fig_predictive_auc(by_level, by_sample, a.tag, a.outdir)

    txt = summary_text(all_rows, by_level, by_sample, a.tag)
    txt_path = os.path.join(a.outdir, f"{a.tag}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(txt)

    print(txt)
    print("\n[analyze] wrote:")
    for p in [p1, p2, p3, p4, txt_path]:
        print("  ", p)


if __name__ == "__main__":
    main()