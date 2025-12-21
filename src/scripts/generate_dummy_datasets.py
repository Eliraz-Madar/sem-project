"""Generate deterministic dummy datasets for MCC and document-level pipelines.

Creates 4 JSONL files under src/data:
  1) mcc_train.jsonl
  2) mcc_dev.jsonl
  3) news_train.jsonl
  4) news_test.jsonl

MCC schema: {"sentence": str, "label": "claim"|"premise"|"other"}
DOC schema: {"doc_id": str, "text": str, "label": "news"|"opinion"} plus optional fields.

Properties:
- Balanced labels in each split (MCC: claim/premise/other; DOC: news/opinion).
- Deterministic content via a fixed seed (configurable).
- Mixed document lengths: short (2-3), medium (5-8), long (10-15) sentences.
- ~10% ambiguous docs per split that include both factual and opinion cues.

Usage (from repo root or src/):
    python -m scripts.generate_dummy_datasets --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MCC_TRAIN = DATA_DIR / "mcc_train.jsonl"
MCC_DEV = DATA_DIR / "mcc_dev.jsonl"
NEWS_TRAIN = DATA_DIR / "news_train.jsonl"
NEWS_TEST = DATA_DIR / "news_test.jsonl"

MCC_LABELS = ("claim", "premise", "other")
DOC_LABELS = ("news", "opinion")

FACTUAL_SNIPPETS = [
    "Officials stated that the measure will take effect next month.",
    "The report included official statistics and quotes from experts.",
    "Sources familiar with the matter confirmed the timeline.",
    "The committee released its summary following the session.",
    "A spokesperson provided a written statement after inquiries.",
    "Government data indicates a year-over-year increase in GDP.",
    "Independent auditors verified the figures in the filing.",
]

OPINION_SNIPPETS = [
    "I believe this decision shows poor judgment.",
    "It seems clear to me that the policy favors elites.",
    "In my view, the government missed an opportunity.",
    "This approach is short-sighted and will cause problems.",
    "One cannot accept this without serious reservations.",
    "Frankly, the proposal is misguided and unrealistic.",
    "I strongly suspect the plan will backfire.",
]

PUBLISHERS = ["NYTimes", "BBC", "Reuters", "TheGuardian", "FoxNews", "WashingtonPost"]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def balanced_labels(labels: Tuple[str, ...], total: int) -> List[str]:
    """Return a list of labels with near-perfect balance (deterministic order)."""
    k = len(labels)
    base = total // k
    rem = total % k
    out: List[str] = []
    for i, lab in enumerate(labels):
        count = base + (1 if i < rem else 0)
        out.extend([lab] * count)
    return out


def make_mcc_sentence(rng: random.Random, label: str, idx: int) -> str:
    # Simple template-based sentence generation per label
    if label == "claim":
        templates = [
            "{topic} is essential for future prosperity.",
            "{topic} will improve outcomes across sectors.",
            "We must adopt {topic} immediately.",
        ]
    elif label == "premise":
        templates = [
            "Studies show {topic} reduces emissions by 20%.",
            "Historical data indicates {topic} correlates with growth.",
            "Experts report {topic} improves reliability in trials.",
        ]
    else:  # other
        templates = [
            "{topic} appears in recent headlines.",
            "There are various discussions around {topic}.",
            "People mention {topic} frequently online.",
        ]
    topics = [
        "renewable energy",
        "digital literacy",
        "public transit",
        "vaccination rates",
        "fiscal policy",
        "urban planning",
    ]
    tpl = templates[idx % len(templates)]
    return tpl.format(topic=topics[idx % len(topics)])


def generate_mcc_split(total: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    labels = balanced_labels(MCC_LABELS, total)
    rows: List[dict] = []
    for i, lab in enumerate(labels):
        sentence = make_mcc_sentence(rng, lab, i)
        rows.append({"sentence": sentence, "label": lab})
    return rows


def choose_length_category(rng: random.Random) -> Tuple[int, int]:
    # short (2-3) ~30%, medium (5-8) ~50%, long (10-15) ~20%
    p = rng.random()
    if p < 0.30:
        return 2, 3
    elif p < 0.80:
        return 5, 8
    else:
        return 10, 15


def make_doc(doc_id: int, label: str, rng: random.Random, ambiguous: bool) -> dict:
    min_len, max_len = choose_length_category(rng)
    num_sent = rng.randint(min_len, max_len)
    sentences: List[str] = []

    def factual_or_opinion_prob(L: str) -> float:
        # base mixture: news mostly factual, opinion mostly opinion
        return 0.85 if L == "news" else 0.25

    mix_prob = factual_or_opinion_prob(label)
    # For ambiguous docs, dampen mix to include both signal types
    if ambiguous:
        mix_prob = 0.5

    for _ in range(num_sent):
        if rng.random() < mix_prob:
            sentences.append(rng.choice(FACTUAL_SNIPPETS))
        else:
            sentences.append(rng.choice(OPINION_SNIPPETS))

    text = " ".join(sentences)
    publisher = rng.choice(PUBLISHERS)
    return {
        "doc_id": str(doc_id),
        "text": text,
        "label": label,
        "publisher": publisher,
    }


def generate_doc_split(total: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    labels = balanced_labels(DOC_LABELS, total)
    ambiguous_n = max(1, int(round(total * 0.10)))
    # Distribute ambiguous flags deterministically across the sequence
    ambiguous_indices = set(range(0, ambiguous_n))

    rows: List[dict] = []
    for i, lab in enumerate(labels, start=1):
        amb = i - 1 in ambiguous_indices
        rows.append(make_doc(i, lab, rng, ambiguous=amb))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic dummy datasets for MCC and doc pipelines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcc-train-n", type=int, default=300)
    parser.add_argument("--mcc-dev-n", type=int, default=90)
    parser.add_argument("--news-train-n", type=int, default=160)
    parser.add_argument("--news-test-n", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # MCC splits
    mcc_train_rows = generate_mcc_split(args.mcc_train_n, seed=args.seed)
    mcc_dev_rows = generate_mcc_split(args.mcc_dev_n, seed=args.seed + 1)
    write_jsonl(MCC_TRAIN, mcc_train_rows)
    write_jsonl(MCC_DEV, mcc_dev_rows)

    # Document splits
    news_train_rows = generate_doc_split(args.news_train_n, seed=args.seed + 2)
    news_test_rows = generate_doc_split(args.news_test_n, seed=args.seed + 3)
    write_jsonl(NEWS_TRAIN, news_train_rows)
    write_jsonl(NEWS_TEST, news_test_rows)

    print(f"Wrote {len(mcc_train_rows)} MCC train sentences to {MCC_TRAIN}")
    print(f"Wrote {len(mcc_dev_rows)} MCC dev sentences to {MCC_DEV}")
    print(f"Wrote {len(news_train_rows)} docs to {NEWS_TRAIN}")
    print(f"Wrote {len(news_test_rows)} docs to {NEWS_TEST}")


if __name__ == "__main__":
    main()
