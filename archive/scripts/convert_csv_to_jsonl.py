"""Convert news_articles.csv to train/test JSONL for the document classification pipeline.

This script:
1. Reads the CSV scraped by scrape_news_rss.py
2. Converts to JSONL format (text, label, source fields)
3. Splits into 80% train, 20% test
4. Saves to src/data/news_train.jsonl and src/data/news_test.jsonl

Usage:
    From src/: python -m scripts.convert_csv_to_jsonl --input data/news_articles.csv --train-output data/news_train.jsonl --test-output data/news_test.jsonl --train-split 0.8
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV articles to train/test JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--train-output", type=Path, default=Path("data/news_train.jsonl"))
    parser.add_argument("--test-output", type=Path, default=Path("data/news_test.jsonl"))
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/test split ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def read_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Read CSV and convert to list of dicts."""
    articles = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or not row.get("full_text", "").strip():
                continue
            articles.append({
                "text": row.get("full_text", "").strip(),
                "label": row.get("category", "fact").strip().lower(),  # 'fact' or 'opinion'
                "title": row.get("title", "").strip(),
                "source": row.get("source", "").strip(),
            })
    return articles


def split_data(articles: List[Dict], train_ratio: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    """Split articles into train and test sets."""
    random.Random(seed).shuffle(articles)
    split_idx = int(len(articles) * train_ratio)
    return articles[:split_idx], articles[split_idx:]


def write_jsonl(articles: List[Dict], output_path: Path) -> None:
    """Write articles to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    
    # Check input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        raise SystemExit(1)
    
    print(f"Reading CSV from {args.input}...")
    articles = read_csv(args.input)
    
    if not articles:
        print("Error: No valid articles found in CSV")
        raise SystemExit(1)
    
    print(f"Loaded {len(articles)} articles")
    
    # Count labels
    fact_count = sum(1 for a in articles if a["label"] == "fact")
    opinion_count = sum(1 for a in articles if a["label"] == "opinion")
    print(f"  - Fact: {fact_count}")
    print(f"  - Opinion: {opinion_count}")
    
    # Split data
    print(f"\nSplitting data (train: {args.train_split*100:.0f}%, test: {(1-args.train_split)*100:.0f}%)...")
    train_articles, test_articles = split_data(articles, args.train_split, args.seed)
    
    train_fact = sum(1 for a in train_articles if a["label"] == "fact")
    train_opinion = sum(1 for a in train_articles if a["label"] == "opinion")
    test_fact = sum(1 for a in test_articles if a["label"] == "fact")
    test_opinion = sum(1 for a in test_articles if a["label"] == "opinion")
    
    print(f"Train set: {len(train_articles)} articles (fact: {train_fact}, opinion: {train_opinion})")
    print(f"Test set: {len(test_articles)} articles (fact: {test_fact}, opinion: {test_opinion})")
    
    # Write JSONL
    print(f"\nWriting JSONL files...")
    write_jsonl(train_articles, args.train_output)
    write_jsonl(test_articles, args.test_output)
    
    print(f"✓ Train: {args.train_output}")
    print(f"✓ Test: {args.test_output}")
    print(f"\nNext steps:")
    print(f"1. Train MCC model: python -m scripts.run_mcc_training --train data/editorials.jsonl --dev data/editorials.jsonl")
    print(f"2. Train document classifier: python -m scripts.run_doc_classification --news-train {args.train_output} --news-test {args.test_output} --mcc-checkpoint checkpoints/mcc_bert.pt")


if __name__ == "__main__":
    main()
