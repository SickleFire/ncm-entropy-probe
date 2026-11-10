# Level-wise Uncertainty and Failure in Next-Concept Modeling: A Sudoku-Hard Pilot Study

This repository contains code, data, figures, and the whitepaper for our pilot study of Level-wise Uncertainty and Failure in Next-Concept Modeling (NCM) on Sudoku-hard.

## Overview

Next-Concept Modeling (NCM) generates text as a 12-level hierarchy of residual VQ codes. We report four observations from a single Sudoku-hard checkpoint (100 puzzles, 84 correct, 16 failures):

- **Constant Coarse Level:** The encoder emits the same Level 4 codes for all 100 puzzles (zero entropy), carrying no puzzle-specific information. Code sequences are not unique: no generated sequence equals the encoder's codes for its puzzle, yet all 84 correct generations decode correctly.
- **Predictive Uncertainty:** Mean entropy at middle levels predicts failure (AUC 0.88 at Level 6, chosen post hoc; Level 0 AUC 0.72). Ranking puzzles by Level 6 entropy puts 30 correct generations first (in-sample).
- **Early Divergence:** All 16 failures first diverge from the encoder's codes within Levels 0–3, and none of the 55 generations that match the encoder through Level 4 fail (Fisher exact p = 4.8 × 10⁻⁷, post hoc cut).
- **Intervention (Forced Prefix):** Forcing the encoder's Level 0–3 codes repairs only 5 of 16 failures and breaks none of the 84 correct generations. Forcing Levels 0–10 repairs all 16. Eleven of 16 failures persist even when the coarse sketch is correct.

## Repository Structure

```text
├── figures/                  # Entropy levels, AUC curves, risk-coverage plots
├── patches/                  # NCM code patches (ncm_stop_at_level.patch)
├── results/                  # Raw probing, forced prefix, and mismatch data
├── scripts/                  # Analysis, probing, forced-prefix probe, plotting scripts
└── WHITEPAPER.md             # Full whitepaper
```

## Quick Start

To reproduce the analysis or inspect probing/analysis scripts:

```bash
python scripts/probe_entropy.py
python scripts/forced_prefix_probe.py
python scripts/mismatch_100.py
python scripts/analyze.py
python scripts/plot_risk_coverage.py
```

## Citation

```bibtex
@article{sarmiento2026levelwise,
  title={Level-wise Uncertainty and Failure in Next-Concept Modeling: A Sudoku-Hard Pilot Study},
  author={Joshua Russel A. Sarmiento},
  year={2026}
}
```
