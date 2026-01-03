"""
Build a labeled FACT vs OPINION dataset from Guardian and All The News CSV files.

Example usage:
  python -m scripts.collect_nyt_news \\
    --guardian-csv path/to/guardian.csv \\
    --allnews-csv path/to/all-the-news-2-1.csv \\
    --out-jsonl dataset.jsonl \\
    --balance

Output schema (JSONL):
  {
    "text": <full article text>,
    "label": "opinion" or "fact",
    "source": "guardian" or "all_the_news",
    "publisher": <publication name or null>,
    "section": <section name or null>,
    "url": <article URL or null>,
    "title": <article title or null>
  }

Labeling rules:
  - Guardian: opinion if pillarName or sectionName contains "Opinion", else fact
  - All The News: opinion if section contains opinion/editorial/op-ed/comment, else fact
  - Both: filter text < 500 chars, deduplicate by SHA1 hash, optionally balance
"""

import os
import sys
import csv
import json
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple
from collections import Counter

try:
    import pandas as pd
except ImportError:
    raise RuntimeError("pandas required: pip install pandas")


MIN_TEXT_LENGTH = 500
OPINION_SECTIONS_ALLNEWS = {"opinion", "editorial", "op-ed", "comment"}


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build labeled FACT/OPINION dataset from Guardian and All The News CSVs"
    )
    parser.add_argument(
        "--guardian-csv",
        type=str,
        help="Path to Guardian CSV (columns: bodyText, pillarName, sectionName, webUrl, title)"
    )
    parser.add_argument(
        "--allnews-csv",
        type=str,
        help="Path to All The News CSV (columns: article, section, publication, title)"
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="dataset.jsonl",
        help="Output JSONL file (default: dataset.jsonl)"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional output CSV file"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample majority class to match minority"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balancing (default: 42)"
    )
    args = parser.parse_args()
    
    if not args.guardian_csv and not args.allnews_csv:
        parser.error("Provide at least one of: --guardian-csv, --allnews-csv")
    
    return args


def normalize_text(text: str) -> str:
    """Normalize text for hashing (strip whitespace, lowercase)."""
    return " ".join(text.lower().split())


def sha1_hash(text: str) -> str:
    """Compute SHA1 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha1(normalized.encode()).hexdigest()


def label_guardian(row: Dict) -> Optional[str]:
    """
    Label Guardian article.
    opinion if pillarName or sectionName contains "Opinion" (case-insensitive).
    """
    pillar = str(row.get("pillarName", "")).lower()
    section = str(row.get("sectionName", "")).lower()
    
    if "opinion" in pillar or "opinion" in section:
        return "opinion"
    return "fact"


def label_allnews(row: Dict) -> Optional[str]:
    """
    Label All The News article.
    opinion if section contains opinion/editorial/op-ed/comment (case-insensitive).
    """
    section = str(row.get("section", "")).lower()
    
    for opinion_keyword in OPINION_SECTIONS_ALLNEWS:
        if opinion_keyword in section:
            return "opinion"
    return "fact"


def process_guardian(csv_path: str) -> List[Dict]:
    """Read and process Guardian CSV."""
    print(f"\n[Guardian] Reading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[Guardian] Loaded {len(df)} rows")
    
    records = []
    skipped_short = 0
    skipped_missing = 0
    
    for _, row in df.iterrows():
        text = str(row.get("bodyText", "")).strip()
        
        if not text:
            skipped_missing += 1
            continue
        
        if len(text) < MIN_TEXT_LENGTH:
            skipped_short += 1
            continue
        
        label = label_guardian(row)
        if not label:
            continue
        
        record = {
            "text": text,
            "label": label,
            "source": "guardian",
            "publisher": str(row.get("publication", "")) or None,
            "section": str(row.get("sectionName", "")) or None,
            "url": str(row.get("webUrl", "")) or None,
            "title": str(row.get("title", "")) or None,
        }
        records.append(record)
    
    print(f"[Guardian] After filtering: {len(records)} records")
    print(f"  └─ Skipped (missing text): {skipped_missing}")
    print(f"  └─ Skipped (text < {MIN_TEXT_LENGTH}): {skipped_short}")
    
    return records


def process_allnews(csv_path: str) -> List[Dict]:
    """Read and process All The News CSV (large file, use chunks)."""
    print(f"\n[All The News] Reading {csv_path}...")
    
    records = []
    skipped_short = 0
    skipped_missing = 0
    chunk_count = 0
    
    # Read in chunks to handle large files
    for chunk in pd.read_csv(csv_path, chunksize=5000, low_memory=False):
        chunk_count += 1
        for _, row in chunk.iterrows():
            text = str(row.get("article", "")).strip()
            
            if not text:
                skipped_missing += 1
                continue
            
            if len(text) < MIN_TEXT_LENGTH:
                skipped_short += 1
                continue
            
            label = label_allnews(row)
            if not label:
                continue
            
            record = {
                "text": text,
                "label": label,
                "source": "all_the_news",
                "publisher": str(row.get("publication", "")) or None,
                "section": str(row.get("section", "")) or None,
                "url": str(row.get("url", "")) or None,
                "title": str(row.get("title", "")) or None,
            }
            records.append(record)
        
        if chunk_count % 5 == 0:
            print(f"  └─ Processed {chunk_count} chunks ({len(records)} records so far)...")
    
    print(f"[All The News] After filtering: {len(records)} records")
    print(f"  └─ Skipped (missing text): {skipped_missing}")
    print(f"  └─ Skipped (text < {MIN_TEXT_LENGTH}): {skipped_short}")
    
    return records


def deduplicate(records: List[Dict]) -> List[Dict]:
    """Remove duplicate texts by SHA1 hash."""
    print(f"\n[Dedup] Deduplicating {len(records)} records...")
    seen = set()
    unique = []
    
    for record in records:
        h = sha1_hash(record["text"])
        if h not in seen:
            seen.add(h)
            unique.append(record)
    
    removed = len(records) - len(unique)
    print(f"[Dedup] Removed {removed} duplicates, {len(unique)} unique remain")
    return unique


def balance_classes(records: List[Dict], seed: int = 42) -> List[Dict]:
    """Downsample majority class to match minority class."""
    import random
    random.seed(seed)
    
    print(f"\n[Balance] Balancing {len(records)} records...")
    
    by_label = {}
    for record in records:
        label = record["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(record)
    
    label_counts = {label: len(recs) for label, recs in by_label.items()}
    print(f"[Balance] Before: {label_counts}")
    
    min_count = min(len(recs) for recs in by_label.values())
    balanced = []
    
    for label, recs in by_label.items():
        if len(recs) > min_count:
            balanced.extend(random.sample(recs, min_count))
        else:
            balanced.extend(recs)
    
    label_counts_after = Counter(r["label"] for r in balanced)
    print(f"[Balance] After: {dict(label_counts_after)}")
    
    return balanced


def write_jsonl(records: List[Dict], out_path: str) -> None:
    """Write records to JSONL file."""
    print(f"\n[Output] Writing JSONL to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[Output] Wrote {len(records)} records to {out_path}")


def write_csv(records: List[Dict], out_path: str) -> None:
    """Write records to CSV file."""
    print(f"\n[Output] Writing CSV to {out_path}...")
    if not records:
        print("[Output] No records to write")
        return
    
    fieldnames = records[0].keys()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[Output] Wrote {len(records)} records to {out_path}")


def main() -> None:
    """Main pipeline."""
    args = get_args()
    
    all_records = []
    
    # Process Guardian
    if args.guardian_csv:
        if not os.path.exists(args.guardian_csv):
            print(f"[Error] Guardian CSV not found: {args.guardian_csv}", file=sys.stderr)
            sys.exit(1)
        guardian_records = process_guardian(args.guardian_csv)
        all_records.extend(guardian_records)
    
    # Process All The News
    if args.allnews_csv:
        if not os.path.exists(args.allnews_csv):
            print(f"[Error] All The News CSV not found: {args.allnews_csv}", file=sys.stderr)
            sys.exit(1)
        allnews_records = process_allnews(args.allnews_csv)
        all_records.extend(allnews_records)
    
    # Deduplicate
    all_records = deduplicate(all_records)
    
    # Balance (optional)
    if args.balance:
        all_records = balance_classes(all_records, seed=args.seed)
    
    # Final label distribution
    label_dist = Counter(r["label"] for r in all_records)
    source_dist = Counter(r["source"] for r in all_records)
    print(f"\n[Summary] Total records: {len(all_records)}")
    print(f"[Summary] By label: {dict(label_dist)}")
    print(f"[Summary] By source: {dict(source_dist)}")
    
    # Write outputs
    write_jsonl(all_records, args.out_jsonl)
    if args.out_csv:
        write_csv(all_records, args.out_csv)
    
    print("\n[Success] Dataset build complete!")


if __name__ == "__main__":
    main()
