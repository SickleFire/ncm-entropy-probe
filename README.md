# Uncertainty at the Commitment Stage Predicts Failure in Coarse-to-Fine Generation

This repository contains the code, data, figures, and whitepaper for our quantitative and causal analysis of Next-Concept Modeling (NCM).

## Overview

Next-Concept Modeling introduces a coarse-to-fine generation paradigm predicting discrete codes hierarchically. In this work, we probe the internal dynamics of NCM during Sudoku reasoning tasks to answer: *Which stage of the refinement process is most diagnostic of final output correctness?*

Key findings:
- **Structural Non-Uniformity:** NCM entropy peaks at coarse and final levels, but collapses to zero at Level 4 (a deterministic structural commitment phase).
- **Predictive Failure:** Mid-level entropy (specifically Level 6) is a strong predictor of final failure (AUC = 0.88), whereas coarse-level entropy is a weak predictor (AUC = 0.72).
- **Causal Brittleness:** Truncating generation at any intermediate level results in absolute failure (0% accuracy), showing zero graceful degradation.

## Repository Structure

```text
├── figures/                  # Generated plots (entropy, AUC, truncation)
├── patches/                  # NCM code patches (stop_at_level.patch)
├── results/                  # Raw probing and truncation data
├── scripts/                  # Analysis, probing, and plotting scripts
└── WHITEPAPER.md             # Full research paper
```

## Quick Start

To reproduce the analysis or inspect probing scripts:

```bash
python scripts/probe_entropy.py
python scripts/probe_truncation.py
python scripts/analyze.py
```

## Citation

```bibtex
@article{sarmiento2026uncertainty,
  title={Uncertainty at the Commitment Stage Predicts Failure in Coarse-to-Fine Generation},
  author={Joshua Russel A. Sarmiento},
  year={2026}
}
```
