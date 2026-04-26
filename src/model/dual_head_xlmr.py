from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


ANCHOR_TOKENS: List[str] = [
    "inshaallah", "mashaallah", "alhamdulillah", "shukar", "umeed",
    "himmat", "dua", "allah", "khuda", "tauba", "aas", "bharosa",
    "yakeen", "hosla", "ummeed", "zindagi", "khushi", "sukoon",
]


@dataclass
class DualHeadConfig:
    model_name: str = "xlm-roberta-base"
    num_labels: int = 4
    anchor_tokens: List[str] = field(default_factory=lambda: list(ANCHOR_TOKENS))
    tokenizer_name: str = ""   # set at runtime to model_name if blank

    def __post_init__(self):
        if not self.tokenizer_name:
            self.tokenizer_name = self.model_name


def _masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states at True positions. Returns zeros where mask is all False."""
    mask_f = mask.unsqueeze(-1).float()          # [B, L, 1]
    summed = (hidden * mask_f).sum(dim=1)        # [B, H]
    count  = mask_f.sum(dim=1).clamp(min=1)      # [B, 1]
    return summed / count                        # [B, H]


class DualHeadXLMRClassifier(nn.Module):
    """
    HLSP v2 — Token-level guided pooling dual-head classifier.

    Head A (Cultural): mean-pools hidden states at cultural anchor token positions
                       (inshaallah, shukar, umeed, allah, dua, ...).
    Head B (English):  mean-pools hidden states at purely ASCII alphabetic token positions
                       (code-mixed English words).
    Fallback: CLS is used for any sample where a head finds no matching tokens.
    Fusion: concat([CLS, head_a, head_b]) → Linear → ReLU → Dropout → 4-class.
    """

    def __init__(self, config: DualHeadConfig):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        H = self.backbone.config.hidden_size

        self._cultural_ids: Set[int] = set()
        self._english_ids: Set[int] = set()
        self._build_token_id_sets(config.tokenizer_name, config.anchor_tokens)

        self.fusion    = nn.Linear(H * 3, H)
        self.dropout   = nn.Dropout(0.2)
        self.classifier = nn.Linear(H, config.num_labels)

    # ------------------------------------------------------------------
    # Vocabulary set builders (called once at __init__)
    # ------------------------------------------------------------------

    def _build_token_id_sets(self, tokenizer_name: str, anchor_tokens: List[str]) -> None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Cultural anchor IDs: encode each anchor word, take first subword piece
        for word in anchor_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                self._cultural_ids.add(ids[0])

        # English token IDs: any vocab token whose surface form is purely ASCII alpha
        vocab = tokenizer.get_vocab()
        for token, tid in vocab.items():
            surface = token.lstrip("▁")   # strip XLM-R leading ▁
            if surface.isalpha() and surface.isascii():
                self._english_ids.add(tid)

    # ------------------------------------------------------------------
    # Per-batch mask builders
    # ------------------------------------------------------------------

    def _cultural_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, L] bool — True at positions matching a cultural anchor token ID."""
        if not self._cultural_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        anchor_t = torch.tensor(
            list(self._cultural_ids), dtype=torch.long, device=input_ids.device
        )
        return (input_ids.unsqueeze(-1) == anchor_t).any(dim=-1)

    def _english_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, L] bool — True at positions matching a purely ASCII alpha token ID."""
        if not self._english_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        eng_t = torch.tensor(
            list(self._english_ids), dtype=torch.long, device=input_ids.device
        )
        # For large vocab sets, chunk to avoid OOM on GPU
        chunk_size = 8192
        result = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(0, len(eng_t), chunk_size):
            chunk = eng_t[i : i + chunk_size]
            result |= (input_ids.unsqueeze(-1) == chunk).any(dim=-1)
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # [B, L, H]
        cls    = hidden[:, 0, :]               # [B, H]

        # Head A — cultural anchor pooling
        c_mask = self._cultural_mask(input_ids)                # [B, L]
        head_a = _masked_mean_pool(hidden, c_mask)             # [B, H]
        empty_a = c_mask.sum(dim=1) == 0                       # [B]
        head_a[empty_a] = cls[empty_a]

        # Head B — English token pooling
        e_mask = self._english_mask(input_ids)                 # [B, L]
        head_b = _masked_mean_pool(hidden, e_mask)             # [B, H]
        empty_b = e_mask.sum(dim=1) == 0                       # [B]
        head_b[empty_b] = cls[empty_b]

        fused  = torch.cat([cls, head_a, head_b], dim=-1)      # [B, H*3]
        fused  = self.dropout(torch.relu(self.fusion(fused)))  # [B, H]
        logits = self.classifier(fused)                        # [B, 4]

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result
