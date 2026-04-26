"""HLSP training: PhoneticNormalizer → DualHeadXLMRClassifier → 4-class.

Run after train_baseline.py. Compares Stage-1-only and full HLSP results.
"""

import csv
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

from .config import ID2LABEL, TRAIN_CLASS_COUNTS, TrainingConfig
from .data_utils import HopeDataset, compute_class_weights, load_split, oversample_minority
from .focal_loss import FocalLoss
from src.normalizer import PhoneticNormalizer
from src.model.dual_head_xlmr import DualHeadConfig, DualHeadXLMRClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    target_names = [ID2LABEL[i] for i in range(4)]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "report": report}


def run_training(cfg: TrainingConfig, texts_train, labels_train, texts_dev, labels_dev, device, tag: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if cfg.oversample_minority:
        texts_train, labels_train = oversample_minority(
            texts_train, labels_train,
            target_label=cfg.oversample_target_label,
            factor=cfg.oversample_factor,
        )
        print(f"[{tag}] After oversampling: {len(texts_train)} train samples")

    train_ds = HopeDataset(texts_train, labels_train, tokenizer, cfg.max_length)
    dev_ds = HopeDataset(texts_dev, labels_dev, tokenizer, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    dual_cfg = DualHeadConfig(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        tokenizer_name=cfg.model_name,
    )
    model = DualHeadXLMRClassifier(dual_cfg).to(device)

    class_weights = compute_class_weights(TRAIN_CLASS_COUNTS, device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    log_path = os.path.join(cfg.results_dir, f"{tag}_train_log.csv")
    best_macro_f1 = 0.0
    best_ckpt = os.path.join(cfg.checkpoint_dir, f"{tag}_best.pt")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_macro_f1", "val_weighted_f1"])

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(out["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            metrics = evaluate(model, dev_loader, device)
            macro_f1 = metrics["macro_f1"]
            weighted_f1 = metrics["weighted_f1"]

            print(
                f"[{tag}] epoch={epoch} loss={avg_loss:.4f} "
                f"macro_f1={macro_f1:.4f} weighted_f1={weighted_f1:.4f}"
            )
            writer.writerow([epoch, f"{avg_loss:.4f}", f"{macro_f1:.4f}", f"{weighted_f1:.4f}"])
            f.flush()

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), best_ckpt)
                print(f"[{tag}] New best checkpoint saved (macro_f1={macro_f1:.4f})")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return evaluate(model, dev_loader, device), model


def main() -> None:
    cfg = TrainingConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HLSP] device={device}")

    normalizer = PhoneticNormalizer()

    raw_train_texts, train_labels, _ = load_split(cfg.train_path, cfg, has_labels=True)
    raw_dev_texts, dev_labels, _ = load_split(cfg.dev_path, cfg, has_labels=True)

    # Stage 1 only: normalized text, dual-head model
    print("\n[HLSP] === Stage 1+2: Normalized text + Dual-Head ===")
    norm_train = [normalizer.normalize(t) for t in raw_train_texts]
    norm_dev = [normalizer.normalize(t) for t in raw_dev_texts]

    final_metrics, _ = run_training(cfg, norm_train, train_labels, norm_dev, dev_labels, device, tag="hlsp")

    print("\n[HLSP] === Final Dev Results ===")
    print(f"  Macro F1:    {final_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {final_metrics['weighted_f1']:.4f}")
    for cls, vals in final_metrics["report"].items():
        if isinstance(vals, dict):
            print(f"  {cls}: f1={vals['f1-score']:.4f} prec={vals['precision']:.4f} rec={vals['recall']:.4f}")

    results_path = os.path.join(cfg.results_dir, "hlsp_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "macro_f1", "weighted_f1", "notes"])
        writer.writerow([
            f"{cfg.model_name}+DualHead+Normalizer",
            f"{final_metrics['macro_f1']:.4f}",
            f"{final_metrics['weighted_f1']:.4f}",
            "normalized roman_urdu, focal loss, 2x Realistic Hope oversample, dual-head",
        ])
    print(f"[HLSP] Results saved to {results_path}")


if __name__ == "__main__":
    main()
