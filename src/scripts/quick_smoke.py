"""Run a tiny end-to-end smoke test in seconds.

What it does (CPU-friendly):
1) Creates a miniature MCC train/dev JSONL in ./tmp_smoke/data.
2) Trains a tiny classifier (prajjwal1/bert-tiny) for 1 epoch.
3) Saves the checkpoint to ./tmp_smoke/checkpoints/mcc_smoke.pt.
4) Runs document-level feature extraction on two toy docs and prints results.

Usage (from repo root or src/):
    python -m scripts.quick_smoke

This is meant only for a quick sanity check; it is not a real model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from mcc.data import LABEL2ID, MCC_LABELS, build_dataloader
from mcc.models import BertMCCClassifier, MCCModelConfig
from doc_clf.features import extract_features_for_corpus

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BASE_DIR / "tmp_smoke"
DATA_DIR = TMP_DIR / "data"
CKPT_DIR = TMP_DIR / "checkpoints"
MCC_TRAIN = DATA_DIR / "mcc_train.jsonl"
MCC_DEV = DATA_DIR / "mcc_dev.jsonl"
DOCS = DATA_DIR / "docs.jsonl"
CKPT_PATH = CKPT_DIR / "mcc_smoke.pt"
MODEL_NAME = "prajjwal1/bert-tiny"  # tiny for fast CPU smoke


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_tiny_mcc_data() -> None:
    train_rows = [
        {"sentence": "Solar power is the future", "label": "claim"},
        {"sentence": "It reduces emissions", "label": "premise"},
        {"sentence": "Coal is cheap", "label": "other"},
    ]
    dev_rows = [
        {"sentence": "Wind is clean", "label": "claim"},
        {"sentence": "It is noisy", "label": "premise"},
    ]
    docs_rows = [
        {"text": "Solar power is the future. It reduces emissions.", "label": "green"},
        {"text": "Coal is cheap. It is noisy.", "label": "brown"},
    ]
    write_jsonl(MCC_TRAIN, train_rows)
    write_jsonl(MCC_DEV, dev_rows)
    write_jsonl(DOCS, docs_rows)


def train_tiny_model(device: torch.device) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = build_dataloader(MCC_TRAIN, tokenizer, batch_size=2, shuffle=True, max_length=64)
    dev_loader = build_dataloader(MCC_DEV, tokenizer, batch_size=2, max_length=64)

    config = MCCModelConfig(pretrained_model_name=MODEL_NAME, num_labels=len(MCC_LABELS))
    model = BertMCCClassifier.from_config(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 1
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps or 1)

    def run_epoch(loader, train: bool) -> float:
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            total_loss += loss.item()
            correct += (preds == batch["labels"]).sum().item()
            total += preds.numel()
            if train:
                loss.backward()
                optimizer.step()
                scheduler.step()
        acc = correct / total if total else 0.0
        return total_loss / max(len(loader), 1), acc

    train_loss, train_acc = run_epoch(train_loader, train=True)
    dev_loss, dev_acc = run_epoch(dev_loader, train=False)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Train loss={train_loss:.4f} acc={train_acc:.4f} | Dev loss={dev_loss:.4f} acc={dev_acc:.4f}")
    print(f"Saved tiny checkpoint to {CKPT_PATH}")


def run_doc_features(device: torch.device) -> None:
    ratios, sequences, labels = extract_features_for_corpus(
        DOCS, CKPT_PATH, model_name=MODEL_NAME, device=device
    )
    print("Doc feature ratios (label -> ratios):")
    for lbl, feats in zip(labels, ratios):
        print(lbl, feats)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    make_tiny_mcc_data()
    train_tiny_model(device)
    run_doc_features(device)


if __name__ == "__main__":
    main()
