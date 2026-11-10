"""
Plot a Risk-Coverage Curve based on mid-level (Level 6) entropy.

Ranks samples by entropy, progressively rejects the highest-entropy samples,
and plots the accuracy of the remaining subset against the coverage.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to entropy probe CSV")
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    # 1. Load the data
    rows = []
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["level"]) == 6: # We use Level 6 as the predictor
                rows.append({
                    "entropy": float(r["H_mean"]),
                    "correct": int(r["correct"])
                })

    if not rows:
        print("No Level 6 data found in CSV.")
        return

    # 2. Sort by entropy (descending, so index 0 is the highest entropy / most uncertain)
    rows.sort(key=lambda x: x["entropy"], reverse=True)
    
    n_total = len(rows)
    accuracies = []
    coverages = []
    
    # 3. Calculate accuracy as we reject the top-N most uncertain samples
    for i in range(n_total):
        # Keep samples from index i to the end
        kept = rows[i:]
        if not kept:
            break
        correct = sum(r["correct"] for r in kept)
        acc = correct / len(kept)
        coverage = len(kept) / n_total
        
        accuracies.append(acc * 100)
        coverages.append(coverage * 100)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot the entropy-based curve (reversed so it goes from 0% coverage to 100%)
    ax.plot(coverages[::-1], accuracies[::-1], 
            marker="", linewidth=2, color="C0", 
            label="Reject by Level 6 Entropy")

    # Plot the random baseline
    baseline_acc = sum(r["correct"] for r in rows) / n_total * 100
    ax.axhline(baseline_acc, linestyle="--", color="gray", alpha=0.7, 
               label=f"Random / Full Generation ({baseline_acc:.1f}%)")

    ax.set_xlabel("Coverage (% of samples kept)")
    ax.set_ylabel("Accuracy on kept samples (%)")
    ax.set_title(f"Risk-Coverage Curve ({a.tag})")
    ax.set_xlim(0, 100)
    ax.set_ylim(min(accuracies) - 5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    path = os.path.join(a.outdir, f"{a.tag}_risk_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

if __name__ == "__main__":
    main()