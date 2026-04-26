"""3-fold stratified cross-validation on training data.

Uses 3 epochs per fold (compute constraint with XLM-RoBERTa).
Reports mean ± std macro F1 across folds.

Run: python -m src.training.cross_validate
"""

import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from .config import TRAIN_CLASS_COUNTS, TrainingConfig
from .data_utils import HopeDataset, compute_class_weights, load_split, oversample_minority
from .focal_loss import FocalLoss
from src.model.dual_head_xlmr import DualHeadConfig, DualHeadXLMRClassifier


CV_EPOCHS = 3
N_FOLDS = 3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(texts, labels, train_idx, val_idx, cfg, device, fold):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_texts, train_labels = oversample_minority(
        train_texts, train_labels,
        target_label=cfg.oversample_target_label,
        factor=cfg.oversample_factor,
    )

    train_ds = HopeDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    val_ds = HopeDataset(val_texts, val_labels, tokenizer, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    dual_cfg = DualHeadConfig(model_name=cfg.model_name, num_labels=cfg.num_labels)
    model = DualHeadXLMRClassifier(dual_cfg).to(device)

    class_weights = compute_class_weights(TRAIN_CLASS_COUNTS, device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * CV_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    best_f1 = 0.0
    for epoch in range(1, CV_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out["logits"], batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].tolist())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(train_loader)
        print(f"  [Fold {fold}] epoch={epoch} loss={avg_loss:.4f} macro_f1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1

    return best_f1


def main():
    cfg = TrainingConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"3-Fold Stratified CV | {CV_EPOCHS} epochs/fold | best macro F1 per fold\n")

    texts, labels, _ = load_split(cfg.train_path, cfg, has_labels=True)
    texts = np.array(texts)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg.seed)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_arr), start=1):
        print(f"\n{'='*45}\nFold {fold}/{N_FOLDS}  (train={len(train_idx)}, val={len(val_idx)})")
        best_f1 = train_one_fold(
            texts.tolist(), labels_arr.tolist(),
            train_idx.tolist(), val_idx.tolist(),
            cfg, device, fold
        )
        fold_scores.append(best_f1)
        print(f"  [Fold {fold}] Best macro F1 = {best_f1:.4f}")

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)

    result = (
        f"\n{'='*45}\n"
        f"3-Fold CV Results (HLSP no_norm, {CV_EPOCHS} epochs/fold)\n"
        f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}\n"
        f"  Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}\n"
    )
    print(result)

    out_path = os.path.join(cfg.results_dir, "cross_validation_results.txt")
    with open(out_path, "w") as f:
        f.write(result)
    print(f"CV results saved to {out_path}")


if __name__ == "__main__":
    main()
