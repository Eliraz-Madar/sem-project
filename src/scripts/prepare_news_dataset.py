"""Prepare a deterministic news_train/news_test JSONL dataset for document-level pipeline.

Behavior:
- If `data/news_train.jsonl` and `data/news_test.jsonl` already exist: do nothing and exit 0.
- Otherwise: generate a small deterministic dataset (default 160 train + 40 test = 200 docs).
- Validate that `checkpoints/mcc_bert.pt` exists; if missing, print a friendly error and exit 1.

This script is Windows-safe and uses pathlib for paths.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "news_train.jsonl"
TEST_FILE = DATA_DIR / "news_test.jsonl"
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "mcc_bert.pt"


def make_doc(doc_id: int, label: str, publisher: str, rng: random.Random) -> dict:
    # generate a document composed of several sentences
    factual_snippets = [
        "Officials stated that the measure will take effect next month.",
        "The report included official statistics and quotes from experts.",
        "Sources familiar with the matter confirmed the timeline.",
        "The committee released its summary following the session.",
        "A spokesperson declined to comment but provided a written statement.",
    ]
    opinion_snippets = [
        "I believe this decision shows poor judgment.",
        "It seems clear to me that the policy favors elites over ordinary people.",
        "In my view, the government missed an opportunity to act differently.",
        "This approach is short-sighted and will cause problems later.",
        "One cannot accept this without serious reservations.",
    ]

    sentences: List[str] = []
    num_sent = rng.randint(3, 8)
    if label == "news":
        # mostly factual sentences with occasional mild commentary
        for _ in range(num_sent):
            if rng.random() < 0.85:
                sentences.append(rng.choice(factual_snippets))
            else:
                sentences.append(rng.choice(opinion_snippets))
    else:
        # opinion pieces: more subjective language
        for _ in range(num_sent):
            if rng.random() < 0.75:
                sentences.append(rng.choice(opinion_snippets))
            else:
                sentences.append(rng.choice(factual_snippets))

    text = " ".join(sentences)
    return {"doc_id": str(doc_id), "text": text, "label": label, "publisher": publisher}


def generate_dataset(seed: int = 42, train_n: int = 160, test_n: int = 40) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    publishers = ["NYTimes", "BBC", "Reuters", "TheGuardian", "FoxNews", "WashingtonPost"]

    rows_train: List[dict] = []
    rows_test: List[dict] = []

    # alternate labels to keep class balance
    for i in range(1, train_n + 1):
        label = "news" if i % 2 == 0 else "opinion"
        pub = rng.choice(publishers)
        rows_train.append(make_doc(i, label, pub, rng))

    for i in range(train_n + 1, train_n + test_n + 1):
        label = "news" if i % 2 == 0 else "opinion"
        pub = rng.choice(publishers)
        rows_test.append(make_doc(i, label, pub, rng))

    with TRAIN_FILE.open("w", encoding="utf-8") as fh:
        for r in rows_train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with TEST_FILE.open("w", encoding="utf-8") as fh:
        for r in rows_test:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows_train)} documents to {TRAIN_FILE}")
    print(f"Wrote {len(rows_test)} documents to {TEST_FILE}")


def main() -> int:
    # Idempotent: if both files exist, do nothing
    if TRAIN_FILE.exists() and TEST_FILE.exists():
        print(f"Found existing {TRAIN_FILE} and {TEST_FILE}; nothing to do.")
        return 0

    # Prefer to have an MCC checkpoint, but proceed regardless
    if not CHECKPOINT.exists():
        print("WARNING: MCC checkpoint not found:", CHECKPOINT)
        print("Proceeding to generate deterministic documents anyway.\n"
              "You can later train MCC (see scripts.run_mcc_training) and rerun document training.")

    # Generate deterministic synthetic dataset
    generate_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
