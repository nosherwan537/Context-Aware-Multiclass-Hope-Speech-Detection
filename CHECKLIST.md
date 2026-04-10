# HLSP Project Checklist

Status keys: [ ] Not started | [~] In progress | [x] Done

## Phase 1 — Setup + Data
- [x] Initialize repository and push to GitHub
- [x] Create project scaffold folders/files
- [ ] Create and activate Python virtual environment
- [ ] Install `requirements.txt`
- [ ] Add raw dataset to `data/raw/`
- [ ] Create stratified splits: `train.csv`, `val.csv`, `test.csv`
- [ ] Confirm split columns: `text`, `label`, `source`
- [ ] Baseline documentation file with target benchmark (~65.2% Macro F1)

## Phase 2 — Normalizer
- [x] Implement text cleaning (URL, emoji, punctuation, whitespace)
- [~] Build seed lexicon (~500 canonical Roman Urdu tokens)
- [x] Implement phonetic mapping + edit-distance mapping
- [x] Add do-not-normalize token whitelist
- [x] Unit tests for 20+ tricky examples
- [x] Measure vocabulary reduction (target 15–25%)
- [~] Manual verification on 100 random samples

## Phase 3 — Modeling
- [ ] Train baseline XLM-R on raw text
- [ ] Train normalized XLM-R (Stage 1 only)
- [ ] Implement dual-head guided attention wrapper
- [ ] Add class-weighted loss + early stopping
- [ ] Save per-epoch logs to CSV
- [ ] Save best checkpoint + config with seed

## Phase 4 — Evaluation
- [ ] Final test evaluation (classification report + confusion matrix)
- [ ] Side-by-side results table (baseline / normalized / full HLSP)
- [ ] Ablation experiments (remove Stage 1 / Head A / Head B)
- [ ] Error analysis of 50 wrong predictions
- [ ] Save all outputs in `results/`

## Phase 5 — Demo + Defense
- [ ] Build local demo app (`demo/app.py`)
- [ ] Show before/after normalization in demo
- [ ] Add confidence score + top attention cues
- [ ] Prepare 12-slide deck
- [ ] Practice 15-minute defense twice

## Immediate Next 3 Actions
- [ ] Create `.venv` and install dependencies
- [ ] Add dataset into `data/raw/` and generate splits
- [ ] Run sanity test: `pytest -q`
