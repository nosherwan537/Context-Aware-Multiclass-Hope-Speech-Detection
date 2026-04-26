"""
Ablation runner — executes all remaining experiment runs for the paper table.
 
Run order:
  3  hlsp_v2            normalizer + dual-head v2 (full)
  4  ablation_no_headA  normalizer + v2, Head A disabled (CLS fallback forced)
  5  ablation_no_headB  normalizer + v2, Head B disabled (CLS fallback forced)
  6  ablation_no_norm   no normalizer + dual-head v2

Usage:
  python -m src.evaluation.ablation                  # all runs
  python -m src.evaluation.ablation --run hlsp_v2    # single run
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

from src.training.config import ID2LABEL, TRAIN_CLASS_COUNTS, TrainingConfig
from src.training.data_utils import HopeDataset, compute_class_weights, load_split, oversample_minority
from src.training.focal_loss import FocalLoss
from src.normalizer import PhoneticNormalizer
from src.model.dual_head_xlmr import DualHeadConfig, DualHeadXLMRClassifier


ABLATION_RUNS = [
    {
        "tag": "hlsp_v2",
        "use_normalizer": True,
        "disable_head_a": False,
        "disable_head_b": False,
        "notes": "normalizer + dual-head v2 (full HLSP)",
    },
    {
        "tag": "ablation_no_headA",
        "use_normalizer": True,
        "disable_head_a": True,
        "disable_head_b": False,
        "notes": "no cultural head (CLS fallback forced for Head A)",
    },
    {
        "tag": "ablation_no_headB",
        "use_normalizer": True,
        "disable_head_a": False,
        "disable_head_b": True,
        "notes": "no English head (CLS fallback forced for Head B)",
    },
    {
        "tag": "ablation_no_norm",
        "use_normalizer": False,
        "disable_head_a": False,
        "disable_head_b": False,
        "notes": "no normalizer + dual-head v2",
    },
]


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
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["labels"].to(device)
            out   = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    target_names = [ID2LABEL[i] for i in range(4)]
    report      = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "report": report}


class _AblatedDualHead(DualHeadXLMRClassifier):
    """Wraps DualHeadXLMRClassifier to force CLS fallback on one or both heads."""

    def __init__(self, config: DualHeadConfig, disable_head_a: bool, disable_head_b: bool):
        super().__init__(config)
        self._disable_head_a = disable_head_a
        self._disable_head_b = disable_head_b

    def forward(self, input_ids, attention_mask=None, labels=None):
        import torch.nn as nn

        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls    = hidden[:, 0, :]

        if self._disable_head_a:
            head_a = cls
        else:
            c_mask = self._cultural_mask(input_ids)
            head_a = self._masked_mean_pool_with_fallback(hidden, c_mask, cls)

        if self._disable_head_b:
            head_b = cls
        else:
            e_mask = self._english_mask(input_ids)
            head_b = self._masked_mean_pool_with_fallback(hidden, e_mask, cls)

        import torch
        fused  = torch.cat([cls, head_a, head_b], dim=-1)
        fused  = self.dropout(torch.relu(self.fusion(fused)))
        logits = self.classifier(fused)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result

    @staticmethod
    def _masked_mean_pool_with_fallback(hidden, mask, cls):
        import torch
        mask_f  = mask.unsqueeze(-1).float()
        summed  = (hidden * mask_f).sum(dim=1)
        count   = mask_f.sum(dim=1).clamp(min=1)
        pooled  = summed / count
        empty   = mask.sum(dim=1) == 0
        pooled[empty] = cls[empty]
        return pooled


def run_ablation(cfg: TrainingConfig, run: dict, raw_train, train_labels,
                 raw_dev, dev_labels, device, summary_writer):

    tag             = run["tag"]
    use_normalizer  = run["use_normalizer"]
    disable_head_a  = run["disable_head_a"]
    disable_head_b  = run["disable_head_b"]

    print(f"\n{'='*60}")
    print(f"[ablation] RUN: {tag}")
    print(f"  normalizer={use_normalizer}  no_headA={disable_head_a}  no_headB={disable_head_b}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    normalizer = PhoneticNormalizer() if use_normalizer else None

    texts_train = [normalizer.normalize(t) for t in raw_train] if normalizer else list(raw_train)
    texts_dev   = [normalizer.normalize(t) for t in raw_dev]   if normalizer else list(raw_dev)

    if cfg.oversample_minority:
        texts_train, train_labels_run = oversample_minority(
            texts_train, list(train_labels),
            target_label=cfg.oversample_target_label,
            factor=cfg.oversample_factor,
        )
        print(f"[{tag}] after oversample: {len(texts_train)} samples")
    else:
        train_labels_run = list(train_labels)

    train_ds = HopeDataset(texts_train, train_labels_run, tokenizer, cfg.max_length)
    dev_ds   = HopeDataset(texts_dev,   list(dev_labels),  tokenizer, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size)

    dual_cfg = DualHeadConfig(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels,
        tokenizer_name=cfg.model_name,
    )
    model = _AblatedDualHead(dual_cfg, disable_head_a, disable_head_b).to(device)

    class_weights = compute_class_weights(TRAIN_CLASS_COUNTS, device)
    loss_fn   = FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)
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
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_b       = batch["labels"].to(device)

                optimizer.zero_grad()
                out  = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(out["logits"], labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            metrics  = evaluate(model, dev_loader, device)
            mf1, wf1 = metrics["macro_f1"], metrics["weighted_f1"]
            print(f"[{tag}] epoch={epoch} loss={avg_loss:.4f} macro_f1={mf1:.4f} weighted_f1={wf1:.4f}")
            writer.writerow([epoch, f"{avg_loss:.4f}", f"{mf1:.4f}", f"{wf1:.4f}"])
            f.flush()

            if mf1 > best_macro_f1:
                best_macro_f1 = mf1
                torch.save(model.state_dict(), best_ckpt)
                print(f"[{tag}] checkpoint saved (macro_f1={mf1:.4f})")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final = evaluate(model, dev_loader, device)

    print(f"\n[{tag}] FINAL  macro_f1={final['macro_f1']:.4f}  weighted_f1={final['weighted_f1']:.4f}")
    for cls, vals in final["report"].items():
        if isinstance(vals, dict):
            print(f"  {cls}: f1={vals['f1-score']:.4f} prec={vals['precision']:.4f} rec={vals['recall']:.4f}")

    gen_hope_f1 = final["report"].get("Generalized Hope", {}).get("f1-score", 0.0)
    summary_writer.writerow([
        tag,
        f"{final['macro_f1']:.4f}",
        f"{final['weighted_f1']:.4f}",
        f"{gen_hope_f1:.4f}",
        run["notes"],
    ])
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None, help="Run a single tag (e.g. hlsp_v2)")
    args = parser.parse_args()

    cfg = TrainingConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}")

    raw_train, train_labels, _ = load_split(cfg.train_path, cfg, has_labels=True)
    raw_dev,   dev_labels,   _ = load_split(cfg.dev_path,   cfg, has_labels=True)

    runs = ABLATION_RUNS
    if args.run:
        runs = [r for r in ABLATION_RUNS if r["tag"] == args.run]
        if not runs:
            print(f"Unknown run tag '{args.run}'. Available: {[r['tag'] for r in ABLATION_RUNS]}")
            sys.exit(1)

    summary_path = os.path.join(cfg.results_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(["tag", "macro_f1", "weighted_f1", "gen_hope_f1", "notes"])

        for run in runs:
            set_seed(cfg.seed)
            run_ablation(cfg, run, raw_train, train_labels, raw_dev, dev_labels, device, writer)
            sf.flush()

    print(f"\n[ablation] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
