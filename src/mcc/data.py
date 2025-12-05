"""Data utilities for the sentence-level argument component classifier (MCC).

The helpers in this module abstract away loading JSONL files, converting the
raw examples into PyTorch ``Dataset`` objects, and creating tokenized
``DataLoader`` instances that the training loop can consume directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

MCC_LABELS: Sequence[str] = ("claim", "premise", "other")
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(MCC_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}


@dataclass
class MCCExample:
    """Container that mirrors the JSONL structure for a single sentence."""

    sentence: str
    label: str
    topic: str | None = None


class MCCDataset(Dataset):
    """Thin ``Dataset`` wrapper around a list of :class:`MCCExample`."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MCC data file not found: {path}")
        self.examples: List[MCCExample] = [
            MCCExample(
                sentence=record["sentence"],
                label=record["label"],
                topic=record.get("topic"),
            )
            for record in _read_jsonl(path)
        ]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        return {"sentence": example.sentence, "label": example.label}


def build_dataloader(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    """Create a tokenizing ``DataLoader`` for MCC data."""

    dataset = MCCDataset(path)

    def collate_fn(batch: Iterable[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        texts = [item["sentence"] for item in batch]
        labels = torch.tensor([LABEL2ID[item["label"].lower()] for item in batch], dtype=torch.long)
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized["labels"] = labels
        return tokenized

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _read_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into memory."""

    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line.strip()) for line in fh if line.strip()]


__all__ = [
    "MCC_LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "MCCDataset",
    "build_dataloader",
]
