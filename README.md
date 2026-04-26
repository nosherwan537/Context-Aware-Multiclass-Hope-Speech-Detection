# Context-Aware Multiclass Hope Speech Detection in Roman Urdu
### Hybrid Lexical-Semantic Pipeline (HLSP) via Culturally-Guided Dual-Head XLM-RoBERTa

> **FAST-NUCES Lahore — Semester Final Project + Research Paper | April 2026**

---

## Overview

Hope speech detection in Urdu has been explored almost exclusively on **Nastaliq (Arabic-script) Urdu**, with datasets confined to shared-task competitions and unavailable to the public. **Roman Urdu** — the Latin-transliterated, code-mixed form used in everyday Pakistani social media — has received virtually no attention in the hope speech literature.

This project makes two primary contributions:

1. **A publicly released Roman Urdu multiclass hope speech dataset** — transliterated from existing Nastaliq corpora using the Qalb transliteration model, manually cleaned, and made available for the first time as an open resource.
2. **HLSP** — a Hybrid Lexical-Semantic Pipeline combining a phonetic normalizer with a culturally-guided dual-head XLM-RoBERTa classifier designed specifically for the orthographic and code-mixing realities of Roman Urdu.

---

## The Dataset Contribution

All prior work on Urdu hope speech (GHaLIB 2026, PolyHope-M 2025, Balouchzahi et al. 2023) operates on standard Nastaliq Urdu. Roman Urdu is a distinct input modality — phonetically noisy, orthographically inconsistent, and code-mixed with English — for which no public labelled hope speech dataset existed.

**What we did:**
- Applied the **Qalb Urdu transliteration model** to convert Nastaliq source text to Roman script
- Performed extensive manual cleaning: normalizing encoding artifacts, handling code-mixed tokens, removing transliteration errors
- Verified label integrity across all four hope classes after script conversion
- Released the cleaned splits publicly under `data/cleaned_data/`

| Split | Samples |
|---|---|
| Train | 3,847 |
| Dev | 1,395 |
| Test (unlabelled) | 1,723 |

**Class distribution (train):**

| Class | Count | % |
|---|---|---|
| Not Hope | 1,697 | 44.1% |
| Generalized Hope | 1,088 | 28.3% |
| Unrealistic Hope | 733 | 19.1% |
| Realistic Hope | 329 | 8.5% |

Input column: `roman_urdu` | Label column: `multiclass`

---

## Model Architecture — HLSP

```
RAW ROMAN URDU TEXT
        |
  [PhoneticNormalizer]   ← Stage 1 (optional — see findings)
        |
  [XLM-RoBERTa Base]
        |
   hidden states
   /    |    \
 CLS  HeadA  HeadB
        |       |
  Cultural   English
  Pooling    Pooling
        \   /
        concat([CLS, HeadA, HeadB])
              |
         Linear(H×3→H) → ReLU → Dropout
              |
         Linear(H→4)
              |
         [4-class output]
```

### Head A — Cultural/Religious Pooling
Mean-pools hidden states at positions of **Urdu cultural anchor tokens**: `inshaallah`, `shukar`, `umeed`, `allah`, `dua`, `himmat`, `alhamdulillah`, and 11 others (18 total). Falls back to CLS if no anchors are present in the input. Designed to capture the religious and motivational language that characterises hope expression in Pakistani social media.

### Head B — Code-Mixed English Pooling
Mean-pools hidden states at positions of **purely ASCII-alphabetic tokens** — the surface form that distinguishes English code-switched words from Urdu tokens in the tokenizer vocabulary. Falls back to CLS if no such tokens are found. Designed to capture English sentiment woven into Urdu sentences.

### Fusion
`concat([CLS, HeadA, HeadB])` → `Linear(3H → H)` → `ReLU` → `Dropout(0.1)` → `Linear(H → 4)`

Both token ID sets are constructed once at `__init__` from the tokenizer vocabulary — no per-forward overhead.

---

## Stage 1 — PhoneticNormalizer

A three-stage Roman Urdu normalizer:
1. **Cleaner** — strips URLs, emojis, punctuation, normalizes whitespace, lowercases
2. **Phonetic mapper** — maps common Roman Urdu phonetic variants to canonical forms (kia→kya, aur→or, etc.)
3. **Levenshtein lexicon** — fuzzy-matches tokens within edit-distance ≤ 2 to a ~500-word canonical Roman Urdu lexicon using `rapidfuzz`

**Normalizer impact on vocabulary:**

| Metric | Value |
|---|---|
| Vocabulary before normalization | 30,349 unique tokens |
| Vocabulary after normalization | 24,572 unique tokens |
| Vocabulary reduction | **19.04%** |
| Rows affected | 86.64% |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 8 |
| Warmup steps | 500 |
| Loss function | Focal Loss (γ=2) + class weights |
| Minority oversampling | 2× Realistic Hope (329 → 658) |
| Max sequence length | 128 |
| Seed | 42 |

---

## Results

### Full Experimental Table

| Model | Macro F1 | Weighted F1 | Gen. Hope F1 | Notes |
|---|---|---|---|---|
| Baseline XLM-R | 0.4791 | 0.5856 | — | Raw Roman Urdu, no modifications |
| HLSP v1 | 0.4819 | 0.5885 | 0.3569 | Normalizer + CLS scalar dual-head |
| HLSP v2 | 0.4752 | 0.5773 | 0.3100 | Normalizer + token-level dual-head |
| ablation\_no\_headA | 0.5414 | 0.6464 | — | Head A disabled, 8 epochs |
| ablation\_no\_headB | 0.5416 | 0.6447 | 0.5057 | Head B disabled, 8 epochs |
| **HLSP (no\_norm)** | **0.5550** | **0.6603** | **0.5251** | **Best — no normalizer, dual-head v2, 8 epochs** |

**Best model improvement over baseline:**
- Macro F1: +7.59 points (+15.8% relative)
- Generalized Hope F1: 0.30 → 0.53 (76% relative improvement)
- Checkpoint: `checkpoints/ablation_no_norm_best.pt`

### Ablation Analysis

The ablation table (all runs at 8 epochs, equal conditions) reveals:

- `no_headA` (0.5414) and `no_headB` (0.5416) are nearly identical — **neither head alone is the bottleneck**
- The full dual-head (both heads active, no normalizer) at 0.5550 outperforms both single-head ablations by ~1.4 macro F1 points — **both heads contribute jointly**
- The normalizer consistently hurts performance when the dual-head is active (see Key Finding below)

---

## Key Finding — Normalizer–Architecture Conflict

The PhoneticNormalizer's cleaner strips all text to `[a-z0-9\s']` (ASCII only). This destroys Head B's English detection signal: Head B identifies code-mixed English tokens by their purely ASCII-alphabetic surface form in the tokenizer vocabulary. After normalization, **all tokens appear ASCII-alphabetic**, eliminating the Urdu/English distinction that Head B relies on. Head B loses its discriminative signal entirely.

This interaction explains why:
- HLSP v2 (normalizer + dual-head) scores **lower** than baseline (0.4752 vs 0.4791)
- Removing the normalizer while keeping the dual-head yields the **best result** (0.5550)
- The normalizer contributes positively as a standalone preprocessing module but conflicts with token-level guided pooling

The PhoneticNormalizer remains a valid standalone contribution for other Roman Urdu NLP tasks where vocabulary standardization is the goal.

---

## Comparison with Prior Work

Direct numerical comparison with existing systems (GHaLIB 2026: 65.2% Urdu multiclass macro F1; Balouchzahi et al. 2023: 48.0%) is **not valid** because:

- All prior systems operate on **Nastaliq (Arabic-script) Urdu** — a different input modality
- GHaLIB uses UrduBERT (trained on Nastaliq) which cannot be applied to Roman Urdu
- Prior datasets were competition-internal and not publicly released

This work operates in a genuinely different and more challenging setting: **Roman Urdu**, for which no prior hope speech baseline existed. The results reported here constitute the **first published baselines** for this task and input modality.

---

## Repository Structure

```
.
├── data/
│   └── cleaned_data/
│       ├── roman_Ur_train_clean.csv     ← 3,847 labelled training samples
│       ├── roman_Ur_dev_clean.csv       ← 1,395 labelled dev samples
│       └── roman_Ur_test_without_labels_clean.csv  ← 1,723 unlabelled test samples
│
├── src/
│   ├── normalizer/                      ← PhoneticNormalizer + lexicon
│   ├── model/
│   │   └── dual_head_xlmr.py           ← DualHeadXLMRClassifier v2
│   ├── training/
│   │   ├── config.py                   ← TrainingConfig, LABEL2ID
│   │   ├── data_utils.py               ← HopeDataset, oversampling, class weights
│   │   ├── focal_loss.py               ← FocalLoss implementation
│   │   ├── train_baseline.py           ← Baseline training loop
│   │   └── train_hlsp.py               ← HLSP training loop
│   └── evaluation/
│       └── ablation.py                 ← All ablation runs
│
├── results/
│   ├── ablation_summary.csv            ← Full ablation table
│   ├── baseline_results.csv
│   ├── hlsp_results.csv
│   └── normalization_metrics.csv
│
├── checkpoints/                         ← Best model checkpoints (.pt)
├── tests/                               ← Unit tests for PhoneticNormalizer
└── demo/app.py                          ← Local demo interface
```

---

## Reproducing Results

```bash
# 1. Setup
python -m venv .venv
source .venv/Scripts/activate       # Windows
pip install -r requirements.txt

# 2. Train baseline
python -m src.training.train_baseline

# 3. Train HLSP (v1 and v2)
python -m src.training.train_hlsp

# 4. Run all ablations
python -m src.evaluation.ablation

# 5. Run a specific ablation
python -m src.evaluation.ablation --run ablation_no_norm
```

---

## Citation

If you use the Roman Urdu hope speech dataset or this codebase, please cite:

```
@misc{hlsp_roman_urdu_2026,
  title   = {Context-Aware Multiclass Hope Speech Detection in Code-Mixed Roman Urdu
             via Hybrid Lexical-Semantic Pipeline},
  author  = {[Authors]},
  year    = {2026},
  note    = {FAST-NUCES Lahore. Dataset transliterated from Nastaliq Urdu
             using the Qalb transliteration model.}
}
```

---

## Acknowledgements

- Qalb Urdu transliteration system — used to convert Nastaliq source data to Roman script
- PolyHope-M 2025 shared task — source annotation schema and class definitions
- XLM-RoBERTa (Conneau et al., 2020) — backbone model
