"""Feature extraction helpers for document-level classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from mcc.data import MCC_LABELS
from mcc.models import BertMCCClassifier, MCCModelConfig


def load_documents(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line.strip()) for line in fh if line.strip()]


def split_into_sentences(text: str) -> List[str]:
    # Naive sentence splitting; real system should use spaCy or nltk.
    sentences = [sent.strip() for sent in text.replace("\n", " ").split(".") if sent.strip()]
    return sentences or [text]


def mcc_tags_for_document(
    model: BertMCCClassifier,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MCC model over each sentence and return logits + predictions."""

    sentences = split_into_sentences(text)
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.cpu()
        preds = logits.argmax(dim=-1)
    return preds, logits


def aggregate_document_features(preds: torch.Tensor) -> Dict[str, float]:
    counts = torch.bincount(preds, minlength=len(MCC_LABELS)).float()
    ratios = counts / counts.sum().clamp(min=1.0)
    return {f"ratio_{label}": ratios[idx].item() for idx, label in enumerate(MCC_LABELS)}


def extract_features_for_corpus(
    docs_path: str | Path,
    model_checkpoint: str | Path,
    model_name: str = "bert-base-uncased",
    device: torch.device | None = None,
) -> Tuple[List[Dict[str, float]], List[List[int]], List[str]]:
    """Extract ratio features and tag sequences for every document."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MCCModelConfig(pretrained_model_name=model_name)
    model = BertMCCClassifier.from_config(config)
    checkpoint = Path(model_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"MCC checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    documents = load_documents(docs_path)
    ratio_features: List[Dict[str, float]] = []
    sequences: List[List[int]] = []
    labels: List[str] = []
    for doc in documents:
        preds, _ = mcc_tags_for_document(model, tokenizer, doc["text"], device)
        ratio_features.append(aggregate_document_features(preds))
        sequences.append(preds.tolist())
        labels.append(doc["label"])
    return ratio_features, sequences, labels


__all__ = [
    "extract_features_for_corpus",
    "load_documents",
    "split_into_sentences",
    "mcc_tags_for_document",
]
