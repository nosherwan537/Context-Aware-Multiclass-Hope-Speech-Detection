# Context-Aware Multiclass Hope Speech Detection (HLSP)

Hybrid Lexical-Semantic Pipeline for code-mixed Roman Urdu hope speech detection.

## Project Structure

- data/raw: original dataset files
- data/processed: normalized/intermediate outputs
- data/splits: train/val/test CSVs
- src/normalizer: phonetic + Levenshtein normalization
- src/model: dual-head XLM-R model wrappers
- src/training: training scripts/config
- src/evaluation: evaluation + ablation utilities
- demo: Streamlit/Gradio demo app
- results: metrics, reports, and tables
- tests: unit tests

## Quick Start

1. Create environment
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies
   - `pip install -r requirements.txt`
3. Add dataset files into `data/raw/`
4. Prepare splits into `data/splits/{train,val,test}.csv` with columns:
   - `text`, `label`, `source`
5. Run baseline training (starter script)
   - `python -m src.training.train_baseline`

## Phase 1 Utilities

- Create stratified splits (70/15/15):
   - `python -m src.training.prepare_splits --input data/raw/your_dataset.csv --label_col label --output_dir data/splits`

## Phase 2 Utilities (Normalizer)

- Normalize and evaluate vocabulary impact:
   - `python -m src.normalizer.evaluate_normalizer --input data/splits/train.csv --output_dir results`

This writes:
- `results/normalization_metrics.csv`
- `results/before_after_samples.csv`
- `results/top_variant_mappings.csv`

## Labels (4-class)

- `realistic_hope`
- `unrealistic_hope`
- `generalized_hope`
- `no_hope`

## Notes

- This repository is scaffolded from the HLSP implementation plan.
- See `CHECKLIST.md` for phase-wise execution tasks.
