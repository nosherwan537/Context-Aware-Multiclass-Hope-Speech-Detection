# HLSP Project Checklist

Status keys: [ ] Not started | [~] In progress | [x] Done

## Phase 1 — Setup + Data
- [x] Initialize repository and push to GitHub
- [x] Create project scaffold folders/files
- [x] Create and activate Python virtual environment (`.venv`)
- [x] Install `requirements.txt` (CUDA-enabled torch via cu128 wheel)
- [x] Real dataset confirmed: `data/cleaned_data/` — train/dev/test pre-split
- [x] Baseline documentation file with target benchmark

## Phase 2 — Normalizer
- [x] Implement text cleaning (URL, emoji, punctuation, whitespace)
- [x] Build seed lexicon (~500 canonical Roman Urdu tokens)
- [x] Implement phonetic mapping + edit-distance mapping
- [x] Add do-not-normalize token whitelist
- [x] Unit tests for 20+ tricky examples
- [x] Measure vocabulary reduction (target 15–25%)
- [~] Manual verification on 100 random samples

## Phase 3 — Modeling
- [x] Implement Focal Loss + class weights (focal_loss.py)
- [x] Implement 2× Realistic Hope oversampling (data_utils.py)
- [x] Train baseline XLM-R — Macro F1: 0.4791
- [x] Train HLSP v1 (normalizer + CLS dual-head) — Macro F1: 0.4819
- [x] Redesign dual-head v2 — token-level guided pooling (src/model/dual_head_xlmr.py)
- [x] Train HLSP v2 (normalizer + token-level dual-head) — Macro F1: 0.4752
- [x] Save best checkpoint + per-epoch logs to CSV

## Phase 3 — Ablation Runs (needed for paper)
- [x] `hlsp_v2` — normalizer + dual-head v2 — Macro F1: 0.4752
- [x] `ablation_no_headA` — Head A disabled — Macro F1: 0.4845
- [x] `ablation_no_headB` — Head B disabled — Macro F1: 0.4820
- [x] `ablation_no_norm` — no normalizer + dual-head v2, 8 epochs — Macro F1: 0.5550 ← BEST

## Key Finding (April 2026)
- Normalizer conflicts with Head B English detection (cleaner strips to ASCII, destroying the Urdu/English distinction)
- Best model: dual-head v2 WITHOUT normalizer, 8 epochs — 0.5550 macro F1, 0.5251 Gen. Hope F1
- +7.6 macro F1 points over baseline (+15.8% relative), Gen. Hope nearly doubled (0.30 → 0.53)
- [ ] Re-run no_headA and no_headB at 8 epochs for fair ablation comparison

## Phase 4 — Evaluation
- [ ] Final results table (baseline / HLSP v1 / HLSP v2 / ablations)
- [ ] Confusion matrix for each model
- [ ] Error analysis of 50 wrong predictions
- [ ] Save all outputs in `results/`

## Phase 5 — Demo + Defense
- [ ] Build local demo app (`demo/app.py`)
- [ ] Show before/after normalization in demo
- [ ] Add confidence score + attention token highlights
- [ ] Prepare 12-slide deck
- [ ] Practice 15-minute defense twice

## Immediate Next Actions
- [x] Fix GPU (CUDA) setup
- [x] Implement dual-head v2 (`src/model/dual_head_xlmr.py`)
- [x] Train HLSP v2 and run ablation experiments
- [ ] Re-run ablation_no_headA and ablation_no_headB at 8 epochs
- [ ] Confusion matrix for best model (ablation_no_norm)
- [ ] Error analysis of 50 wrong predictions
- [ ] Build demo app
- [ ] Prepare defense slides
