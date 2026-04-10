from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class DualHeadConfig:
    model_name: str = "xlm-roberta-base"
    num_labels: int = 4
    cultural_boost: float = 1.15
    mixed_boost: float = 1.10


class DualHeadXLMRClassifier(nn.Module):
    """Starter dual-head wrapper for HLSP.

    This keeps architecture lightweight as planned.
    """

    def __init__(self, config: DualHeadConfig):
        super().__init__()
        self.config = config
        self.backbone = AutoModel.from_pretrained(config.model_name)
        hidden = self.backbone.config.hidden_size

        self.cultural_proj = nn.Linear(hidden, hidden)
        self.mixed_proj = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden * 2, config.num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask=None, labels=None) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]

        head_a = torch.tanh(self.cultural_proj(cls)) * self.config.cultural_boost
        head_b = torch.tanh(self.mixed_proj(cls)) * self.config.mixed_boost

        fused = self.dropout(torch.cat([head_a, head_b], dim=-1))
        logits = self.classifier(fused)

        result = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)
        return result
