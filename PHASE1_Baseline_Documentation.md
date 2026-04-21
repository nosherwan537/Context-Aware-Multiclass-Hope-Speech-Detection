# Phase 1 Baseline Documentation

## Target Benchmark
- Target baseline to beat: **65.2% Macro F1** (multi-class)
- Project target: **>75% Macro F1** with HLSP

## Evaluation Metrics (Fixed)
- Macro F1 (primary)
- Weighted F1
- Per-class Precision / Recall / F1
- Confusion Matrix

## Baseline Protocol
- Model: `xlm-roberta-base`
- Data: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`
- Text input: raw text (no normalizer)
- Seed: 42

## Reporting Template
- Save baseline metrics to `results/baseline_results.csv` with:
  - `model`
  - `macro_f1`
  - `weighted_f1`
  - `notes`

## Current Status
- Dataset splits are present and versioned.
- Training scaffold exists at `src/training/train_baseline.py`.
- Final baseline run and recorded numbers pending.
