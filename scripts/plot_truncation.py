import os
import re
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_results(path):
    """Parse the truncation output file into (label, accuracy) pairs."""
    results = []
    with open(path) as f:
        for line in f:
            # Lines look like:
            #   stop_after_level_0: 0/15 (0.0%)
            #   full: 13/15 (86.7%)
            m = re.match(r"(stop_after_level_\d+|full):\s+\d+/\d+\s+\(([\d.]+)%\)", line.strip())
            if m:
                key, acc = m.group(1), float(m.group(2))
                if key == "full":
                    label = "Full\nGeneration"
                else:
                    lvl = key.split("_")[-1]
                    label = f"Stop after\nLevel {lvl}"
                results.append((label, acc))
    # Sort by level (full goes last)
    def sort_key(item):
        if item[0] == "Full\nGeneration":
            return 999
        return int(item[0].split("Level ")[-1])
    results.sort(key=sort_key)
    return results


def plot(results, tag, outdir):
    labels = [r[0] for r in results]
    accs = [r[1] for r in results]

    # Color: all truncations red, full green
    colors = ["#d62728" if "Full" not in l else "#2ca02c" for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, accs, color=colors, width=0.6)

    # annotate each bar with the accuracy
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{tag}: accuracy vs. truncation point (Sudoku-hard)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"{tag}_truncation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to truncation results .txt")
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    results = parse_results(a.input)
    if not results:
        print(f"[plot_truncation] no results found in {a.input}")
        return

    print("[plot_truncation] parsed:")
    for label, acc in results:
        print(f"  {label.replace(chr(10), ' ')}: {acc}%")

    path = plot(results, a.tag, a.outdir)
    print(f"[plot_truncation] wrote {path}")


if __name__ == "__main__":
    main()