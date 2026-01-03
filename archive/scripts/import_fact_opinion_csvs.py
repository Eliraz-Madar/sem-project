"""Import two CSVs (fact + opinion) and produce train/test JSONL for Part B.

- Detects text/title columns automatically (overridable via CLI)
- Labels rows from the first CSV as `fact` and the second as `opinion`
- Shuffles and splits into train/test

Usage (from src/):
  python -m scripts.import_fact_opinion_csvs \
    --fact-csv "C:/path/to/fact.csv" \
    --opinion-csv "C:/path/to/opinion.csv" \
    --train-out data/news_train.jsonl \
    --test-out data/news_test.jsonl

Optional:
  --text-col TEXTCOL   # if your CSV column for article text has a custom name
  --title-col TITLECOL # optional title column name
  --train-split 0.8    # default 0.8
  --seed 42            # default 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

TEXT_CANDIDATES = ["text", "full_text", "article", "body", "content"]
TITLE_CANDIDATES = ["title", "headline"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import fact/opinion CSVs and write train/test JSONL")
    p.add_argument("--fact-csv", type=Path, required=True)
    p.add_argument("--opinion-csv", type=Path, required=True)
    p.add_argument("--text-col", type=str, default=None, help="Text column name (applied to both files)")
    p.add_argument("--title-col", type=str, default=None, help="Title column name (applied to both files)")
    p.add_argument("--train-out", type=Path, default=Path("data/news_train.jsonl"))
    p.add_argument("--test-out", type=Path, default=Path("data/news_test.jsonl"))
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    # Let pandas infer encoding; fallback to python engine if needed
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False, engine="python")


def to_records(df: pd.DataFrame, label: str, text_col: Optional[str], title_col: Optional[str]) -> List[Dict]:
    # Auto-detect columns if not provided
    t_col = text_col or detect_col(df, [c.lower() for c in TEXT_CANDIDATES])
    ttl_col = title_col or detect_col(df, [c.lower() for c in TITLE_CANDIDATES])

    if not t_col or t_col not in df.columns:
        raise ValueError(f"Could not detect text column in {list(df.columns)}; pass --text-col explicitly")

    out: List[Dict] = []
    for _, row in df.iterrows():
        text = str(row.get(t_col, "")).strip()
        if not text:
            continue
        title = str(row.get(ttl_col, "")).strip() if ttl_col else ""
        out.append({
            "text": text,
            "label": label,
            "title": title,
            "source": label,  # optional marker
        })
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    fact_df = load_csv(args.fact_csv)
    op_df = load_csv(args.opinion_csv)

    fact_rows = to_records(fact_df, "fact", args.text_col, args.title_col)
    op_rows = to_records(op_df, "opinion", args.text_col, args.title_col)

    all_rows = fact_rows + op_rows
    if not all_rows:
        print("No rows produced. Check column names with --text-col/--title-col.")
        raise SystemExit(1)

    # Shuffle + split
    rng = pd.Series(range(len(all_rows))).sample(frac=1.0, random_state=args.seed).tolist()
    split_idx = int(len(all_rows) * float(args.train_split))
    train_idx = set(rng[:split_idx])

    train_rows = [all_rows[i] for i in train_idx]
    test_rows = [all_rows[i] for i in range(len(all_rows)) if i not in train_idx]

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.test_out, test_rows)

    print(f"Wrote train: {args.train_out} ({len(train_rows)})")
    print(f"Wrote test : {args.test_out} ({len(test_rows)})")
    print("Next:")
    print(f"  python -m scripts.run_doc_classification --news-train {args.train_out} --news-test {args.test_out} --mcc-checkpoint checkpoints/mcc_bert.pt")


if __name__ == "__main__":
    main()
