"""Full evaluation of best checkpoint on dev set.

Outputs:
  results/full_metrics.txt   — per-class precision/recall/F1/accuracy
  results/confusion_matrix.png
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.training.config import ID2LABEL, TrainingConfig
from src.training.data_utils import HopeDataset, load_split
from src.model.dual_head_xlmr import DualHeadConfig, DualHeadXLMRClassifier


BEST_CHECKPOINT = "checkpoints/ablation_no_norm_best.pt"
LABEL_NAMES = [ID2LABEL[i] for i in range(4)]


def load_model(checkpoint_path: str, cfg: TrainingConfig, device):
    dual_cfg = DualHeadConfig(model_name=cfg.model_name, num_labels=cfg.num_labels)
    model = DualHeadXLMRClassifier(dual_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def save_confusion_matrix(labels, preds, output_path: str):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Best Model (HLSP no_norm) — Confusion Matrix (Dev Set)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    cfg = TrainingConfig()
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dev_texts, dev_labels, _ = load_split(cfg.dev_path, cfg, has_labels=True)
    dev_ds = HopeDataset(dev_texts, dev_labels, tokenizer, cfg.max_length)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    model = load_model(BEST_CHECKPOINT, cfg, device)
    print(f"Loaded checkpoint: {BEST_CHECKPOINT}")

    labels, preds = get_predictions(model, dev_loader, device)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, target_names=LABEL_NAMES, digits=4)

    output = (
        f"Model: HLSP no_norm (DualHead XLM-RoBERTa, no normalizer, 8 epochs)\n"
        f"Checkpoint: {BEST_CHECKPOINT}\n"
        f"Evaluated on: {cfg.dev_path} ({len(labels)} samples)\n"
        f"\n{'='*55}\n"
        f"Overall Accuracy : {accuracy:.4f}\n"
        f"Macro F1         : {macro_f1:.4f}\n"
        f"Weighted F1      : {weighted_f1:.4f}\n"
        f"\nPer-Class Report:\n{report}"
    )

    print(output)

    metrics_path = os.path.join(cfg.results_dir, "full_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(output)
    print(f"Metrics saved to {metrics_path}")

    cm_path = os.path.join(cfg.results_dir, "confusion_matrix.png")
    save_confusion_matrix(labels, preds, cm_path)


if __name__ == "__main__":
    main()
