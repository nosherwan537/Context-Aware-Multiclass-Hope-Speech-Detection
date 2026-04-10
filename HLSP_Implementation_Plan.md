# ENGINEERING IMPLEMENTATION PLAN
## Context-Aware Multiclass Hope Speech Detection in Code-Mixed Roman Urdu via Hybrid Lexical-Semantic Pipeline (HLSP)

| Project Type | Team Size | Total Duration | Version |
|---|---|---|---|
| Semester Final Project | 2 Members | ~15 Weeks | v1.0 — April 2026 |

---

## 1. Project Overview

### 1.1 Problem Statement

Current state-of-the-art NLP models (e.g., GHaLIB 2026, XLM-RoBERTa) treat Roman Urdu as a "black box" — they feed raw, phonetically noisy text directly into Transformer architectures without any language-aware preprocessing. This creates three compounding failures:

- **Embedding Dilution:** `kia`, `kiya`, `kya` are treated as three distinct tokens, forcing the model to learn the same semantic concept multiple times
- **Cultural Marker Blindness:** Religious and hope-bearing tokens like `InshaAllah`, `Shukar`, `Alhamdulillah` are weighted the same as stop-words
- **Code-Mixed Confusion:** English sentiments woven into Urdu sentences lose their syntactic context in a general-purpose multilingual model

### 1.2 Proposed Solution — HLSP

The Hybrid Lexical-Semantic Pipeline (HLSP) introduces a two-stage approach:

- **Stage 1 — Phonetic-Aware Normalization Layer:** Collapses spelling variations using Levenshtein Distance and phonetic mapping before the text ever reaches the Transformer
- **Stage 2 — Dual-Head Attention Transformer:** A modified fine-tuned XLM-RoBERTa where one head specializes in Urdu cultural/religious markers and another captures English code-mixed patterns

### 1.3 Research Gap Being Addressed

> **The Normalization Gap (Core Research Contribution)**
>
> - While 2025–2026 SOTA models achieve ~95% binary accuracy, multi-class hope detection drops to ~65%.
> - No existing system applies phonetic normalization + culturally-guided attention specifically for Roman Urdu hope speech.
> - HLSP bridges this gap by acting as a "White Box" pipeline — standardizing orthography BEFORE semantic learning.

### 1.4 Expected Outcomes

- A working end-to-end ML pipeline for multiclass Roman Urdu hope speech detection
- Improved multi-class F1 score over GHaLIB/XLM-R baseline (targeting >75% from 65.2%)
- A phonetic normalization module reusable for other Roman Urdu NLP tasks
- A well-documented codebase and a demo interface for the defense presentation

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
RAW TEXT → [Stage 1: Phonetic Normalizer] → [Stage 2: Dual-Head Transformer] → HOPE CLASS
```

**Stage 1 Sub-components:**
- Text Cleaner (noise, URLs, special chars)
- Phonetic Mapper (Soundex / Double Metaphone adapted for Urdu phonemes)
- Levenshtein Distance Lexicon (edit-distance threshold clustering)
- Canonical Form Converter (kia → kya, umeed → umeed)

**Stage 2 Sub-components:**
- Tokenizer (XLM-RoBERTa BPE tokenizer)
- Head A — Urdu Cultural Attention (InshaAllah, Shukar, Alhamdulillah markers)
- Head B — Code-Mixed English Attention (English tokens + switch-point detection)
- Classification Head (Realistic Hope / Unrealistic Hope / Generalized Hope / No Hope)

### 2.2 Model Architecture Details

| Component | Technology / Method | Purpose |
|---|---|---|
| Base Model | XLM-RoBERTa (fine-tuned) | Multilingual Transformer backbone |
| Normalization | Levenshtein Distance + Phonetic Lexicon | Collapse spelling variants to canonical form |
| Attention Head A | Guided Self-Attention (cultural tokens) | Detect Urdu hope markers (InshaAllah etc.) |
| Attention Head B | Guided Self-Attention (English tokens) | Handle English code-switched segments |
| Classification | Softmax 4-class head | Realistic / Unrealistic / Generalized / No Hope |
| Dataset | Posi-Vox-2024 + custom annotations | Training and evaluation corpus |
| Evaluation | Weighted F1, Macro F1, Confusion Matrix | Multi-class performance measurement |

### 2.3 Hope Classes — Classification Schema

| Class Label | Description | Example (Roman Urdu) |
|---|---|---|
| Realistic Hope | Grounded, achievable positive expectation | "Mehnat karo, InshaAllah kamyaab ho ge" |
| Unrealistic Hope | Wishful, ungrounded optimism | "Lottery lagegi aur sab theek ho jaye ga" |
| Generalized Hope | Broad community/religious hope | "Shukar hai, Pakistan ka bhala hoga" |
| No Hope / Other | Neutral, negative, or sarcastic text | "Bohot umeed hai aap se" (sarcastic) |

---

## 3. Implementation Phases

The project is organized into 5 phases spanning approximately 15 weeks. Each phase builds directly on the previous and has clearly defined tasks and deliverables.

---

### PHASE 1 — Research, Setup & Dataset Preparation
**Timeline:** Weeks 1–3 | **Deliverable:** Cleaned, annotated dataset + dev environment

#### 3.1.1 Environment Setup
- Create a shared GitHub repository with a clear branch strategy (main, dev, feature branches)
- Set up a shared Python virtual environment — use `requirements.txt` for reproducibility
- Configure Jupyter notebooks or VS Code with Git integration
- Tools: Python 3.10+, PyTorch, HuggingFace Transformers, scikit-learn, pandas, numpy

#### 3.1.2 Literature Review & Baseline Documentation
- Read and summarize the 4 core papers: GHaLIB 2026, PolyHope-M 2025, PAS 2026, PeerJ 2025
- Document the exact accuracy numbers you are trying to beat (65.2% multi-class baseline)
- Define your evaluation metrics: Weighted F1, Macro F1, per-class Precision/Recall

#### 3.1.3 Dataset Acquisition & Annotation
- Download Posi-Vox-2024 dataset and examine its structure
- Identify and handle class imbalance — check distribution of the 4 hope classes
- Manual annotation: review 200–300 ambiguous samples as a team to agree on labeling guidelines
- Split dataset: 70% train / 15% validation / 15% test — ensure stratified splitting
- Save all splits as CSV files with consistent column names: `text`, `label`, `source`

> **Phase 1 Quality Gate**
> - ✓ GitHub repo initialized with README, .gitignore, requirements.txt
> - ✓ Dataset split files saved and version-controlled
> - ✓ Baseline XLM-R performance documented (you need this to prove your improvement)
> - ✓ Annotation guidelines written and agreed upon by both team members

---

### PHASE 2 — Phonetic-Aware Normalization Module
**Timeline:** Weeks 4–6 | **Deliverable:** Working normalizer with evaluation report

#### 3.2.1 Text Cleaning Pipeline
- Strip URLs, emojis, HTML tags, and repeated punctuation
- Normalize whitespace and handle mixed encoding issues (common in scraped Roman Urdu)
- Lowercase all text — Roman Urdu is case-insensitive in practice

#### 3.2.2 Phonetic Mapping Layer
- Build a Roman Urdu phoneme table — map common letters to Urdu sound equivalents
  - Examples: k/q → k sound, aa/a → long a, ee/i → long i
- Adapt Double Metaphone algorithm for Urdu phoneme set (or use Soundex as fallback)
- Create a Python class `PhoneticNormalizer` with a `normalize(text)` method

#### 3.2.3 Levenshtein Distance Lexicon
- Build a seed lexicon of ~500 canonical Roman Urdu words (most common hope/emotion vocabulary)
  - Examples: umeed, InshaAllah, himmat, shukar, kamyabi, mushkil
- Implement edit-distance clustering: any word within distance ≤ 2 of a canonical form gets mapped to it
- Use `rapidfuzz` Python library for efficient Levenshtein computation at scale
- Handle edge cases: very short words (< 3 chars) should not be normalized aggressively

#### 3.2.4 Normalization Evaluation
- Measure vocabulary reduction ratio: how many unique tokens before vs. after normalization
- Manually verify 100 random samples — check that normalization did not change meaning
- Check that sarcasm and emotional intensity tokens are preserved (do not over-normalize)

> **Phase 2 Quality Gate**
> - ✓ Normalizer reduces vocabulary size by at least 15–25%
> - ✓ Manual verification shows no meaning-altering changes in 100-sample check
> - ✓ `normalize()` function is modular and independently testable
> - ✓ Unit tests written for at least 20 known tricky cases (kia/kya, umeed/umid etc.)

---

### PHASE 3 — Dual-Head Attention Model Development
**Timeline:** Weeks 7–10 | **Deliverable:** Trained HLSP model + evaluation results

#### 3.3.1 Baseline Model (XLM-RoBERTa)
- Fine-tune vanilla XLM-RoBERTa on your dataset WITHOUT the normalizer — this is your official baseline
- Record Weighted F1 and Macro F1 — these numbers go directly into your defense presentation
- Use HuggingFace Trainer API for clean, reproducible training
- Training config: batch size 16, learning rate 2e-5, epochs 5, warmup steps 500

#### 3.3.2 Normalization-Enhanced Model
- Run your entire dataset through the `PhoneticNormalizer` first, then fine-tune XLM-RoBERTa on the normalized text
- Compare results against the raw baseline — document the delta in F1 scores
- This proves the value of Stage 1 independently before adding Dual-Head Attention

#### 3.3.3 Dual-Head Attention Implementation
- Extend the XLM-RoBERTa model class to add guided attention weighting
- **Head A (Cultural Head):** Create a list of Urdu cultural/religious anchor tokens — these get an attention weight multiplier during forward pass
  - Anchor token list: `InshaAllah, MashAllah, Shukar, Alhamdulillah, Himmat, Umeed, Tauba`
- **Head B (Code-Mixed Head):** Detect token-language using `langdetect` or a simple Unicode heuristic — English tokens feed into this head's weighted attention
- Implement as a custom PyTorch Module that wraps the HuggingFace model
- Keep the modification lightweight — do not rebuild the entire architecture from scratch

#### 3.3.4 Training Strategy
- Use class-weighted loss function (`CrossEntropyLoss` with weights) to handle class imbalance
- Implement early stopping with patience=3 on validation Macro F1
- Save model checkpoints after every epoch — keep the best checkpoint by validation F1
- Log training metrics to a CSV file: `epoch, train_loss, val_loss, val_f1`

> **Phase 3 Quality Gate**
> - ✓ Baseline XLM-R results documented (this is your comparison benchmark)
> - ✓ HLSP model shows improvement on multi-class Macro F1 vs. baseline
> - ✓ Confusion matrix generated — shows which hope class benefits most from HLSP
> - ✓ Training is reproducible (set random seed, save config file with all hyperparameters)

---

### PHASE 4 — Evaluation, Analysis & Error Study
**Timeline:** Weeks 11–12 | **Deliverable:** Evaluation report + error analysis document

#### 3.4.1 Full Model Evaluation
- Run final evaluation on the held-out test set (never touched during training)
- Generate: Classification Report (precision, recall, F1 per class), Confusion Matrix, ROC Curves
- Compare three models side-by-side: Baseline XLM-R, Normalized XLM-R, Full HLSP

| Model | Weighted F1 | Macro F1 | Hope-class F1 | Notes |
|---|---|---|---|---|
| Baseline XLM-RoBERTa | TBD | ~65% | TBD | Raw text, no modifications |
| + Normalization only | TBD | TBD | TBD | Stage 1 only |
| Full HLSP (Target) | TBD | >75% | TBD | Stage 1 + Dual-Head |

#### 3.4.2 Error Analysis
- Examine the 50 most confidently wrong predictions — understand failure patterns
- Categorize errors: sarcasm misclassification, code-switching edge cases, rare hope expressions
- Document these as a "Limitations" section — this shows maturity and will impress your defense panel

#### 3.4.3 Ablation Study
- Test what happens when you REMOVE parts of your system:
  - Without normalization: shows how much Stage 1 contributes
  - Without Head A (Cultural): shows how much cultural markers help
  - Without Head B (Code-Mixed): shows the value of language-aware attention
- Present this as a table in your defense — panels love ablation studies as proof of engineering rigor

> **Phase 4 Quality Gate**
> - ✓ Test set evaluation complete — final numbers locked in
> - ✓ Ablation results table ready for presentation slides
> - ✓ At least 20 error cases documented and categorized
> - ✓ All results saved to a `results/` folder in the repository

---

### PHASE 5 — Demo Interface & Defense Preparation
**Timeline:** Weeks 13–15 | **Deliverable:** Live demo + final presentation

#### 3.5.1 Demo Interface
- Build a simple Gradio or Streamlit web interface — no complex front-end needed
- Input: a Roman Urdu text box
- Output: predicted hope class + confidence score + which tokens were high-attention (using LIME or attention visualization)
- Show a "before normalization" vs "after normalization" comparison panel — this visually explains Stage 1
- Host locally for the defense demo (no deployment needed for semester project)

#### 3.5.2 Presentation Structure (Recommended Slide Order)

| Slide # | Section | Key Point |
|---|---|---|
| 1 | Title Slide | Project name, team, date |
| 2 | The Problem | Roman Urdu black box — show embedding dilution visually |
| 3 | Research Gap | Gap table: Sarcasm / Intensity / Orthography / Context |
| 4 | Proposed Solution | HLSP pipeline diagram — Stage 1 → Stage 2 |
| 5 | Stage 1 Deep Dive | Phonetic normalizer with kia/kya example |
| 6 | Stage 2 Deep Dive | Dual-Head Attention diagram — Urdu Head + English Head |
| 7 | Dataset & Setup | Posi-Vox-2024 stats, class distribution |
| 8 | Results | 3-model comparison table + confusion matrix |
| 9 | Ablation Study | What happens when you remove each component |
| 10 | Live Demo | Switch to browser — run 3–4 live examples |
| 11 | Limitations & Future Work | Sarcasm, intensity granularity, real-time deployment |
| 12 | Conclusion | One-paragraph summary of contribution |

#### 3.5.3 Killer Question Preparation

**Q: "Why XLM-RoBERTa and not mBERT?"**
> A: XLM-RoBERTa was trained on larger and more diverse multilingual data including Roman scripts; mBERT is weaker on low-resource code-mixed languages.

**Q: "Doesn't normalization lose emotional intensity? (e.g., pleeease vs. please)"**
> A: Stage 1 normalizes lexical tokens but the Dual-Head Attention in Stage 2 still processes the original sentence structure and code-mixed patterns. Emotional intensity markers survive in the semantic layer.

**Q: "How is this different from just using a spell-checker?"**
> A: A spell-checker enforces English or Urdu (Nastaliq) rules. Our normalizer uses phonetic distance specific to Roman Urdu phonology — it understands that kia and kya are phonetically identical, which no English spell-checker would know.

> **Phase 5 Quality Gate**
> - ✓ Demo runs without errors on at least 10 test inputs
> - ✓ All 12 presentation slides complete and reviewed by both team members
> - ✓ Code repository is clean: remove debug prints, add docstrings, write a README
> - ✓ Practice full 15-minute defense run-through at least twice before the actual defense

---

## 4. Master Timeline

| Week | Phase | Key Tasks | Milestone |
|---|---|---|---|
| 1 | Phase 1 | Repo setup, literature review, paper summaries | Dev environment ready |
| 2 | Phase 1 | Dataset download, initial EDA, class distribution analysis | Dataset understood |
| 3 | Phase 1 | Annotation, dataset splitting, baseline documentation | Clean dataset splits saved |
| 4 | Phase 2 | Text cleaner implementation and testing | Cleaner module done |
| 5 | Phase 2 | Phonetic mapper + seed lexicon construction | Phonetic engine draft |
| 6 | Phase 2 | Levenshtein lexicon + normalizer evaluation | Normalizer complete |
| 7 | Phase 3 | Baseline XLM-R fine-tuning on raw data | Baseline F1 recorded |
| 8 | Phase 3 | Normalized XLM-R fine-tuning, Stage 1 impact measured | Stage 1 contribution proven |
| 9 | Phase 3 | Dual-Head Attention implementation | Custom model architecture done |
| 10 | Phase 3 | Full HLSP training, hyperparameter tuning | Best model checkpoint saved |
| 11 | Phase 4 | Final test set evaluation, confusion matrix | Final numbers locked |
| 12 | Phase 4 | Ablation study + error analysis | Error analysis report done |
| 13 | Phase 5 | Gradio/Streamlit demo interface | Live demo working |
| 14 | Phase 5 | Presentation slides + practice run #1 | Slides v1 complete |
| 15 | Phase 5 | Final review, practice run #2, code cleanup | **DEFENSE READY** |

---

## 5. Technology Stack

| Category | Tool / Library | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | All ML and scripting |
| Deep Learning | PyTorch | 2.x | Model training and custom layers |
| NLP / Transformers | HuggingFace Transformers | 4.x | XLM-RoBERTa loading, fine-tuning, tokenizer |
| Edit Distance | rapidfuzz | latest | Fast Levenshtein computation for normalization |
| Language Detection | langdetect | latest | Identify English vs Urdu tokens for Dual-Head |
| Data Handling | pandas, numpy | latest | Dataset processing and manipulation |
| Evaluation | scikit-learn | latest | F1, confusion matrix, classification report |
| Visualization | matplotlib, seaborn | latest | Training curves, confusion matrices |
| Demo Interface | Gradio or Streamlit | latest | Defense live demo |
| Version Control | Git + GitHub | latest | Collaboration and code history |
| Experiment Tracking | CSV logging (or W&B) | latest | Track training runs |
| Notebooks | Jupyter Lab | latest | Exploration and EDA |

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| Dual-Head implementation too complex to finish on time | Medium | High | Implement simpler attention weighting first (token-list multiplication). Save full custom head for stretch goal. |
| HLSP does not improve over baseline | Low–Medium | High | Document every intermediate result. Even null results are valid if methodology is sound. Ablation will show partial contributions. |
| Dataset too small or highly imbalanced | Medium | Medium | Use class-weighted loss + data augmentation (back-translation via Google Translate API for minority classes). |
| Normalization over-collapses meaning | Low | Medium | Hard-code a "do not normalize" whitelist for emotionally significant tokens. Manual verification at Phase 2. |
| Running out of compute / slow training | Medium | Medium | Use Google Colab Pro or Kaggle free GPU. Reduce batch size. Use gradient accumulation. |
| Team coordination issues | Low | Medium | Weekly sync meeting. Use GitHub Issues to track tasks. Document decisions in a shared Notion/Google Doc. |

---

## 7. Recommended Repository Structure

```
hlsp-hope-detection/
│
├── data/
│    ├── raw/               ← Original Posi-Vox-2024 files
│    ├── processed/         ← After normalization
│    └── splits/            ← train.csv, val.csv, test.csv
│
├── src/
│    ├── normalizer/        ← PhoneticNormalizer class + lexicon files
│    ├── model/             ← Dual-Head model definition (PyTorch)
│    ├── training/          ← Training scripts + config files
│    └── evaluation/        ← Eval scripts, metrics, confusion matrix plotter
│
├── notebooks/
│    ├── 01_EDA.ipynb
│    ├── 02_normalization_analysis.ipynb
│    └── 03_results_analysis.ipynb
│
├── demo/
│    └── app.py             ← Gradio / Streamlit demo
│
├── results/
│    ├── baseline_results.csv
│    ├── hlsp_results.csv
│    └── ablation_table.csv
│
├── requirements.txt
└── README.md
```

---

## 8. Definition of Done — Project Completion Checklist

### Codebase
- ✓ PhoneticNormalizer module with unit tests
- ✓ Dual-Head Attention model implemented and documented
- ✓ Training script with reproducible seed and saved hyperparameter config
- ✓ Evaluation script that generates all metrics from a single command
- ✓ GitHub repo is clean, has a README, and all notebooks are re-runnable

### Results
- ✓ Baseline results documented
- ✓ HLSP outperforms baseline on Macro F1 (or contribution of each component is clearly shown via ablation)
- ✓ Ablation table shows individual component contributions
- ✓ Error analysis identifies at least 3 failure pattern categories

### Presentation & Demo
- ✓ 12-slide deck covering all required sections
- ✓ Live Gradio/Streamlit demo runs on local machine without errors
- ✓ Both team members have rehearsed responses to the 3 killer questions
- ✓ Full 15-minute mock defense completed at least twice

---

*HLSP Hope Speech Detection — Engineering Implementation Plan | Version 1.0 | April 2026*
