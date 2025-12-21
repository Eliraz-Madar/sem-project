"""Diagnostics for the document classification pipeline (Part B).

This script inspects dataset fields, sentence splitting, MCC checkpoint loading,
MCC inference outputs, and doc-level feature extraction without training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from doc_clf.features import (
    aggregate_document_features,
    split_into_sentences,
)
from mcc.data import MCC_LABELS
from mcc.models import BertMCCClassifier, MCCModelConfig

TEXT_CANDIDATES = ["text", "content", "body", "article", "document"]
LABEL_CANDIDATES = ["label", "category", "y", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug document classification pipeline (no training)")
    parser.add_argument("--news-train", type=Path, required=True)
    parser.add_argument("--news-test", type=Path, required=True)
    parser.add_argument("--mcc-checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-docs", type=int, default=5)
    parser.add_argument("--max-sents", type=int, default=12)
    return parser.parse_args()


def read_first_n(path: Path, limit: int) -> Tuple[List[dict], set[str]]:
    docs: List[dict] = []
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs.append(obj)
            keys.update(obj.keys())
            if len(docs) >= limit:
                break
    return docs, keys


def detect_field(keys: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in keys:
            return cand
    return None


def label_distribution(docs: List[dict], label_field: Optional[str]) -> Counter:
    dist: Counter = Counter()
    if not label_field:
        return dist
    for doc in docs:
        if label_field in doc:
            val = str(doc[label_field]).strip().lower()
            dist[val] += 1
    return dist


def shorten(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_dataset_info(name: str, docs: List[dict], keys: set[str], text_field: Optional[str], label_field: Optional[str]) -> None:
    print(f"[{name}] Loaded docs: {len(docs)}")
    print(f"[{name}] Keys seen: {sorted(keys)}")
    if not text_field:
        print(f"[{name}] WARNING: No text field found in {TEXT_CANDIDATES}")
    else:
        print(f"[{name}] Using text field: '{text_field}'")
    if not label_field:
        print(f"[{name}] WARNING: No label field found in {LABEL_CANDIDATES}")
    else:
        dist = label_distribution(docs, label_field)
        print(f"[{name}] Label distribution (subset): {dict(dist)}")


def load_mcc_model(checkpoint: Path, model_name: str, device: torch.device) -> Tuple[Optional[BertMCCClassifier], Optional[str]]:
    try:
        config = MCCModelConfig(pretrained_model_name=model_name)
        model = BertMCCClassifier.from_config(config)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device).eval()
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def run_mcc_on_sentences(model: BertMCCClassifier, tokenizer, sentences: List[str], device: torch.device) -> Optional[torch.Tensor]:
    if not sentences:
        return None
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        preds = outputs.logits.argmax(dim=-1).cpu()
    return preds


def pad_sequences(sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    if max_len == 0:
        return torch.zeros((len(sequences), 0), dtype=torch.long), lengths
    padded = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        if seq:
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Early checks for required files
    missing: List[str] = []
    if not args.news_train.exists():
        missing.append(f"train dataset not found: {args.news_train}")
    if not args.news_test.exists():
        missing.append(f"test dataset not found: {args.news_test}")
    if not args.mcc_checkpoint.exists():
        missing.append(f"MCC checkpoint not found: {args.mcc_checkpoint}")

    if missing:
        print("Missing required inputs:")
        for msg in missing:
            print(f"- {msg}")
        print("Suggestions:")
        print("- If you need demo data, run: python -m scripts.prepare_news_dataset")
        print("- If MCC checkpoint is missing, train via: python -m scripts.run_mcc_training --train data/editorials.jsonl --dev data/editorials.jsonl")
        raise SystemExit(1)

    train_docs, train_keys = read_first_n(args.news_train, args.max_docs)
    test_docs, test_keys = read_first_n(args.news_test, args.max_docs)
    all_keys = train_keys | test_keys

    text_field = detect_field(all_keys, TEXT_CANDIDATES)
    label_field = detect_field(all_keys, LABEL_CANDIDATES)

    print("=== Dataset overview ===")
    print_dataset_info("train", train_docs, train_keys, text_field, label_field)
    print_dataset_info("test", test_docs, test_keys, text_field, label_field)

    print("\n=== Sentence splitting ===")
    for name, docs in (("train", train_docs), ("test", test_docs)):
        print(f"-- {name} --")
        for idx, doc in enumerate(docs):
            text = str(doc.get(text_field, "")) if text_field else ""
            label_val = doc.get(label_field, "?") if label_field else "?"
            sentences = split_into_sentences(text)
            print(f"doc {idx}: label={label_val} chars={len(text)} sentences={len(sentences)}")
            for sent in sentences[:3]:
                print(f"    {shorten(sent)}")
            if len(sentences) <= 1:
                print("    WARNING: sentence count <= 1")

    print("\n=== MCC checkpoint & inference ===")
    model, load_err = load_mcc_model(args.mcc_checkpoint, args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if load_err:
        print(f"Failed to load MCC checkpoint: {load_err}")
    else:
        for name, docs in (("train", train_docs), ("test", test_docs)):
            print(f"-- {name} --")
            for idx, doc in enumerate(docs):
                text = str(doc.get(text_field, "")) if text_field else ""
                sentences = split_into_sentences(text)[: args.max_sents]
                preds = run_mcc_on_sentences(model, tokenizer, sentences, device)
                if preds is None:
                    print(f"doc {idx}: no sentences to tag")
                    continue
                counts = torch.bincount(preds, minlength=len(MCC_LABELS)).tolist()
                seq_labels = [MCC_LABELS[p] for p in preds.tolist()]
                print(f"doc {idx}: tag_counts={{'claim': {counts[0]}, 'premise': {counts[1]}, 'other': {counts[2]}}}")
                print(f"         seq(first {args.max_sents}): {seq_labels[: args.max_sents]}")
                if len(set(seq_labels)) == 1:
                    print("         WARNING: all predictions are the same for this doc")

    print("\n=== Feature extraction (subset) ===")
    ratio_features: List[Dict[str, float]] = []
    sequences: List[List[int]] = []
    if model is not None:
        for doc in train_docs:
            text = str(doc.get(text_field, "")) if text_field else ""
            sentences = split_into_sentences(text)[: args.max_sents]
            preds = run_mcc_on_sentences(model, tokenizer, sentences, device)
            if preds is None:
                sequences.append([])
                ratio_features.append({f"ratio_{lbl}": 0.0 for lbl in MCC_LABELS})
                continue
            ratio_features.append(aggregate_document_features(preds))
            sequences.append(preds.tolist())

        padded, lengths = pad_sequences(sequences)
        ratio_shape = (len(ratio_features), len(MCC_LABELS)) if ratio_features else (0, 0)
        print(f"ratio feature shape: {ratio_shape}")
        print(f"sequence padded shape: {tuple(padded.shape)}")
        if lengths.numel():
            lengths_np = lengths.numpy()
            print(
                f"sequence lengths: min={lengths_np.min()} mean={lengths_np.mean():.2f} max={lengths_np.max()}"
            )
        else:
            print("sequence lengths: none")

        if ratio_features:
            print(f"first ratio vector: {ratio_features[0]}")
        if sequences:
            print(f"first sequence length: {len(sequences[0])}")

        # basic sanity warnings
        any_zero_seq = any(len(seq) == 0 for seq in sequences)
        if any_zero_seq:
            print("WARNING: some documents produced empty sequences")
        if ratio_features:
            vals = [v for feat in ratio_features for v in feat.values()]
            if not vals or all(v == 0.0 for v in vals):
                print("WARNING: ratio features are all zeros")
            if any(torch.isnan(torch.tensor(vals))):
                print("WARNING: NaN detected in ratio features")

    print("\n=== What to do next ===")
    if not text_field:
        print(f"- Add/rename a text field to one of {TEXT_CANDIDATES} in your JSONL.")
    if not label_field:
        print(f"- Add/rename a label field to one of {LABEL_CANDIDATES} in your JSONL.")
    if load_err:
        print(f"- MCC failed to load. Ensure checkpoint exists at: {args.mcc_checkpoint} (error: {load_err})")
    print("- If sentence counts are 0 or 1, replace the naive split function or pre-split sentences in data.")
    print("- If all MCC predictions collapse to one tag, re-check the checkpoint or input text cleanliness.")
    print("- If ratio features are empty/NaN, inspect sentence splitting and MCC inference outputs above.")


if __name__ == "__main__":
    main()
