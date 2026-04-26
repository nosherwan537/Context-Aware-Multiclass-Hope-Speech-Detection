# Dual-Head Redesign Plan — Real Token-Level Guided Attention

## Why the Current Implementation Falls Short

The v1 dual-head model applies two linear projections to the **same CLS token**:

```
CLS → tanh(Linear_A(CLS)) × 1.15   ← Head A
CLS → tanh(Linear_B(CLS)) × 1.10   ← Head B
```

Both heads receive identical information. The 1.15/1.10 multipliers are scalar constants —
they add no token-level selectivity. The model cannot distinguish which tokens are
culturally significant vs. code-mixed English.

**Result:** Generalized Hope recall improved (+35% relative) from normalization alone,
but the dual-head contributed no additional signal.

---

## v2 Architecture — Token-Level Guided Pooling

Instead of projecting CLS twice, each head pools over a **different subset of token positions**
from the full sequence hidden states.

```
                    XLM-RoBERTa backbone
                          │
              token hidden states [B, L, H]
                    ┌─────┴──────┐
                    │            │
             CLS[:,0,:]     all tokens
                 [B,H]       [B,L,H]
                    │       ┌────┴────┐
                    │   Head A     Head B
                    │  cultural   English
                    │  pooling    pooling
                    │   [B,H]     [B,H]
                    └────┬────────┬───┘
                         │ concat │
                    [B, H×3]
                         │
                    Linear → H
                    Dropout
                    Linear → 4 classes
```

### Head A — Cultural Anchor Pooling
- At runtime, look up which token IDs correspond to anchor tokens
  (`inshaallah`, `allah`, `shukar`, `umeed`, `himmat`, `dua`, etc.)
- Find all positions in the sequence where `input_ids` matches any anchor token ID
- **Mean-pool** the hidden states at those positions → `[B, H]`
- **Fallback:** if no anchor tokens found in a sample, use CLS for that sample

### Head B — English Token Pooling
- Detect English tokens via a vocabulary heuristic: token IDs whose decoded string
  is purely ASCII alphabetic (a-z) are English
- Mean-pool hidden states at those positions → `[B, H]`
- **Fallback:** if no English tokens found, use CLS for that sample

### Fusion
```
fused = concat([cls, head_a_pool, head_b_pool])   # [B, H×3]
fused = dropout(relu(linear_fusion(fused)))         # [B, H]
logits = classifier(fused)                          # [B, 4]
```

---

## Implementation Steps

### Step 1 — Build anchor token ID lookup
- After loading the tokenizer, encode each anchor token string
- Store as a `set` of token IDs for O(1) lookup during forward pass
- Handle subword tokenization: a word like `inshaallah` may split into 2–3 subword pieces;
  use the first piece ID as the anchor signal

### Step 2 — Build English token ID set
- Iterate over the tokenizer vocabulary (~250k tokens for XLM-R)
- Keep IDs where `tokenizer.convert_ids_to_tokens(id).strip('▁')` is purely ASCII alpha
- Cache this set at model init — do not recompute per forward pass

### Step 3 — Implement masked mean-pool helper
```python
def masked_mean_pool(hidden, mask):
    # hidden: [B, L, H], mask: [B, L] bool
    # returns: [B, H], falls back to zeros where mask is all False
    mask_f = mask.unsqueeze(-1).float()          # [B, L, 1]
    summed = (hidden * mask_f).sum(dim=1)        # [B, H]
    count  = mask_f.sum(dim=1).clamp(min=1)      # [B, 1]
    return summed / count
```

### Step 4 — Rewrite DualHeadXLMRClassifier.forward()
```python
def forward(self, input_ids, attention_mask, labels=None):
    out = self.backbone(input_ids, attention_mask)
    hidden = out.last_hidden_state          # [B, L, H]
    cls    = hidden[:, 0, :]               # [B, H]

    # Head A: cultural anchor positions
    cultural_mask = self._cultural_mask(input_ids)   # [B, L] bool
    head_a = masked_mean_pool(hidden, cultural_mask) # [B, H]
    # fallback: where mask is empty, use CLS
    empty_a = cultural_mask.sum(dim=1) == 0          # [B]
    head_a[empty_a] = cls[empty_a]

    # Head B: English token positions
    english_mask = self._english_mask(input_ids)     # [B, L] bool
    head_b = masked_mean_pool(hidden, english_mask)  # [B, H]
    empty_b = english_mask.sum(dim=1) == 0
    head_b[empty_b] = cls[empty_b]

    fused  = torch.cat([cls, head_a, head_b], dim=-1)  # [B, H×3]
    fused  = self.dropout(torch.relu(self.fusion(fused)))
    logits = self.classifier(fused)
    ...
```

### Step 5 — Update DualHeadConfig
- Add `anchor_tokens: List[str]` (defaults to ANCHOR_TOKENS from lexicon)
- Add `tokenizer_name: str` (needed to build vocab sets at init)

### Step 6 — Update train_hlsp.py
- Pass tokenizer name into DualHeadConfig
- No other changes needed — data pipeline is unchanged

---

## Expected Impact

| Class | Why it should improve |
|---|---|
| Generalized Hope | Head A will pool over religious markers — direct signal for this class |
| Realistic Hope | Head B will capture English action verbs mixed in (mehnat + "work hard") |
| Unrealistic Hope | Both heads — code-mixed wishful English + religious framing |
| Not Hope | Less dilution of CLS from irrelevant head projections |

---

## Files to Change

| File | Change |
|---|---|
| `src/model/dual_head_xlmr.py` | Full rewrite of forward() + anchor/English mask builders |
| `src/training/train_hlsp.py` | Pass tokenizer to DualHeadConfig |
| `src/training/config.py` | No change needed |

---

## Experiment Protocol

Run in this order to build the ablation table for the paper:

| Run | Normalizer | Head A | Head B | Tag |
|---|---|---|---|---|
| 1 (done) | ✗ | ✗ | ✗ | `baseline` |
| 2 (done) | ✓ | v1 (CLS×scalar) | v1 | `hlsp_v1` |
| 3 | ✓ | v2 (anchor pool) | v2 (english pool) | `hlsp_v2` |
| 4 (ablation) | ✓ | ✗ (CLS fallback) | v2 | `ablation_no_headA` |
| 5 (ablation) | ✓ | v2 | ✗ (CLS fallback) | `ablation_no_headB` |
| 6 (ablation) | ✗ | v2 | v2 | `ablation_no_norm` |

Runs 4–6 generate the ablation table that proves each component's individual contribution.

---

## Current Results (for paper baseline table)

| Model | Macro F1 | Weighted F1 | Generalized Hope F1 | Notes |
|---|---|---|---|---|
| Baseline XLM-R | 0.4791 | 0.5856 | 0.3009 | raw roman_urdu, no modifications |
| HLSP v1 | 0.4819 | 0.5885 | 0.3569 | normalizer + CLS dual-head |
| HLSP v2 | TBD | TBD | TBD | normalizer + token-level dual-head |

*Dataset: roman_Ur_train_clean.csv (3847 train, 1395 dev). Focal Loss γ=2, 2× Realistic Hope oversample.*

---

*HLSP Dual-Head Redesign Plan | v2.0 | April 2026*
