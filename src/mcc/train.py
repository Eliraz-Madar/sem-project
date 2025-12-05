"""Training script for the MCC sentence-level classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import build_dataloader
from .models import BertMCCClassifier, MCCModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MCC classifier")
    parser.add_argument("--train", type=Path, default=Path("data/mcc_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/mcc_dev.jsonl"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mcc_bert.pt"))
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def train_epoch(model, dataloader: DataLoader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            total_loss += loss.item()
            correct += (preds == batch["labels"]).sum().item()
            total += preds.numel()
    acc = correct / total if total else 0.0
    return total_loss / max(len(dataloader), 1), acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = build_dataloader(args.train, tokenizer, batch_size=args.batch_size, shuffle=True)
    dev_loader = build_dataloader(args.dev, tokenizer, batch_size=args.batch_size)

    config = MCCModelConfig(pretrained_model_name=args.model, num_labels=3)
    model = BertMCCClassifier.from_config(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * max(args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    best_acc = 0.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}")
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    # Minimal smoke test with dummy data if files do not exist.
    tmp_train = Path("data/mcc_train.jsonl")
    tmp_dev = Path("data/mcc_dev.jsonl")
    tmp_train.parent.mkdir(parents=True, exist_ok=True)
    if not tmp_train.exists():
        tmp_train.write_text('{"sentence": "The sky is blue", "label": "claim"}\n', encoding="utf-8")
    if not tmp_dev.exists():
        tmp_dev.write_text('{"sentence": "It might rain", "label": "other"}\n', encoding="utf-8")
    main()
