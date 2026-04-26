import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .config import LABEL2ID, TRAIN_CLASS_COUNTS, TrainingConfig


class HopeDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_split(path: str, cfg: TrainingConfig, has_labels: bool = True):
    df = pd.read_csv(path)
    texts = df[cfg.text_col].fillna("").tolist()
    labels = None
    if has_labels:
        labels = df[cfg.label_col].map(LABEL2ID).tolist()
    return texts, labels, df


def oversample_minority(
    texts: List[str],
    labels: List[int],
    target_label: int,
    factor: int,
) -> Tuple[List[str], List[int]]:
    """Duplicate samples of target_label by factor and shuffle."""
    minority_idx = [i for i, l in enumerate(labels) if l == target_label]
    extra_texts = [texts[i] for i in minority_idx] * (factor - 1)
    extra_labels = [labels[i] for i in minority_idx] * (factor - 1)
    combined_texts = texts + extra_texts
    combined_labels = labels + extra_labels
    paired = list(zip(combined_texts, combined_labels))
    random.shuffle(paired)
    t, l = zip(*paired)
    return list(t), list(l)


def compute_class_weights(counts: List[int], device: torch.device) -> torch.Tensor:
    counts_arr = np.array(counts, dtype=np.float32)
    total = counts_arr.sum()
    weights = total / (len(counts_arr) * counts_arr)
    return torch.tensor(weights, dtype=torch.float32).to(device)
