"""Quick benchmark of multiple document-level classifiers on a small subset.

- Loads a JSONL dataset with fields like {"text": ..., "label": ...}.
- Subsamples to --max-docs and does a train/dev split.
- Compares Logistic Regression, Linear SVM, and Random Forest over TF-IDF.
- Optionally, if an MCC checkpoint and doc data exist, compares a ratio-feature model.

Usage (from repo root or src/):
    python -m scripts.compare_doc_models

This is intended to run in seconds on CPU as a sanity check.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from doc_clf.features import extract_features_for_corpus
from mcc.data import MCC_LABELS

DEFAULT_DATA_PATH = Path("data/news_train.jsonl")
DEFAULT_MCC_CKPT = Path("checkpoints/mcc_bert.pt")
DEFAULT_MODEL_NAME = "bert-base-uncased"


def load_jsonl(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    if not path.exists():
        return texts, labels
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj and "label" in obj:
                texts.append(str(obj["text"]))
                labels.append(str(obj["label"]))
    return texts, labels


def maybe_stratified_split(texts: List[str], labels: List[str], test_size: float, seed: int):
    try:
        return train_test_split(texts, labels, test_size=test_size, random_state=seed, stratify=labels)
    except ValueError:
        return train_test_split(texts, labels, test_size=test_size, random_state=seed)


def evaluate_model(name: str, model, X_train, X_dev, y_train, y_dev, show_report: bool = False):
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)
    acc = accuracy_score(y_dev, preds)
    macro_f1 = f1_score(y_dev, preds, average="macro")
    if show_report:
        print(f"\n{name} classification report:\n{classification_report(y_dev, preds)}")
    return {"model": name, "acc": acc, "macro_f1": macro_f1}


def build_tfidf_logreg(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("clf", LogisticRegression(max_iter=200, random_state=seed)),
        ]
    )


def build_tfidf_svm() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("clf", LinearSVC()),
        ]
    )


def build_tfidf_rf(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ]
    )


def maybe_ratio_model(docs_path: Path, mcc_checkpoint: Path, model_name: str, seed: int):
    if not docs_path.exists() or not mcc_checkpoint.exists():
        return None, None, None
    try:
        ratios, _, labels = extract_features_for_corpus(docs_path, mcc_checkpoint, model_name=model_name)
    except FileNotFoundError:
        return None, None, None
    feature_matrix = np.array([[feat[f"ratio_{label}"] for label in MCC_LABELS] for feat in ratios])
    return feature_matrix, labels, LogisticRegression(max_iter=200, random_state=seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare lightweight document classifiers on a small subset")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mcc-checkpoint", type=Path, default=DEFAULT_MCC_CKPT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--show-report", action="store_true", help="Print full classification report per model")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.random_seed)

    texts, labels = load_jsonl(args.data_path)
    if not texts:
        print(f"Dataset not found or empty at {args.data_path}; please create it first.")
        return

    if len(texts) > args.max_docs:
        indices = list(range(len(texts)))
        random.Random(args.random_seed).shuffle(indices)
        keep = set(indices[: args.max_docs])
        texts = [t for i, t in enumerate(texts) if i in keep]
        labels = [l for i, l in enumerate(labels) if i in keep]

    X_train, X_dev, y_train, y_dev = maybe_stratified_split(texts, labels, args.test_size, args.random_seed)

    models = [
        ("LogReg-TFIDF", build_tfidf_logreg(args.random_seed)),
        ("LinearSVC-TFIDF", build_tfidf_svm()),
        ("RF-TFIDF", build_tfidf_rf(args.random_seed)),
    ]

    # Optional ratio-based model (if MCC checkpoint and docs are available)
    ratio_matrix, ratio_labels, ratio_clf = maybe_ratio_model(args.data_path, args.mcc_checkpoint, args.model_name, args.random_seed)
    if ratio_matrix is not None and len(ratio_labels) >= 4:
        try:
            X_train_r, X_dev_r, y_train_r, y_dev_r = train_test_split(
                ratio_matrix, ratio_labels, test_size=args.test_size, random_state=args.random_seed, stratify=ratio_labels
            )
        except ValueError:
            X_train_r, X_dev_r, y_train_r, y_dev_r = train_test_split(
                ratio_matrix, ratio_labels, test_size=args.test_size, random_state=args.random_seed
            )
        models.append(("Ratio-LogReg", ratio_clf))
    else:
        X_train_r = X_dev_r = y_train_r = y_dev_r = None

    results = []
    print(f"==== Model comparison (max_docs={args.max_docs}) on {args.data_path} ====")
    for name, model in tqdm(models, desc="models", leave=False):
        if name.startswith("Ratio"):
            if X_train_r is None:
                continue
            res = evaluate_model(name, model, X_train_r, X_dev_r, y_train_r, y_dev_r, show_report=args.show_report)
        else:
            res = evaluate_model(name, model, X_train, X_dev, y_train, y_dev, show_report=args.show_report)
        results.append(res)

    print("Model\tAcc\tMacro-F1")
    for res in results:
        print(f"{res['model']}\t{res['acc']:.3f}\t{res['macro_f1']:.3f}")
    print(f"Used {len(texts)} documents from {args.data_path}")


if __name__ == "__main__":
    main()
