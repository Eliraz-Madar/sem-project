"""Convert Webis editorials annotations to MCC JSONL.

Expected input
- Directory: src/corpus-webis-editorials-16/annotated-txt/split-by-portal-final/
- Contains portal folders (aljazeera/, foxnews/, guardian/) with .txt files.
- Each .txt line looks like: ``0\tlabel\ttext`` where ``label`` may be
  continued, assumption, anecdote, statistics, testimony, no-unit, par-sep, etc.

Output
- JSONL at src/data/editorials.jsonl with objects {"sentence": str, "label": str}
- Labels normalized to lowercase claim/premise/other to match mcc/data.py

Usage
- From repo root (or src/): ``python -m scripts.convert_webis_to_jsonl``
- Optional args: ``--input DIR`` to point to another corpus root; ``--output FILE``

Assumptions
- "continued" rows inherit the most recent non-continued label.
- Unknown labels fall back to "other".
- Lines with label "par-sep" are skipped; "title" becomes "other".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "corpus-webis-editorials-16" / "annotated-txt" / "split-by-portal-final"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "editorials.jsonl"

# Map raw labels to MCC buckets.
CLAIM_LIKE = {"claim", "assumption", "common-ground", "major-claim", "conclusion", "stance"}
PREMISE_LIKE = {"premise", "anecdote", "statistics", "testimony", "example", "reason", "background", "fact"}
SKIP_LABELS = {"par-sep"}
TITLE_LABELS = {"title"}


def normalize_label(raw_label: str, last_label: str | None) -> Tuple[str | None, str | None]:
    """Return (effective_label, normalized_bucket).

    - continued inherits the last_label.
    - par-sep rows are skipped (return (None, None)).
    - title rows map to other.
    - unknown labels map to other.
    """

    if raw_label in SKIP_LABELS:
        return None, None

    effective = last_label if raw_label == "continued" else raw_label
    if effective is None:
        effective = "other"

    if effective in TITLE_LABELS:
        bucket = "other"
    elif effective in CLAIM_LIKE:
        bucket = "claim"
    elif effective in PREMISE_LIKE:
        bucket = "premise"
    elif effective == "no-unit":
        bucket = "other"
    else:
        bucket = "other"

    return effective, bucket


def iter_annotations(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (sentence, label) pairs from a single annotated .txt file."""

    last_label: str | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            _, raw_label, *text_parts = parts
            text = "\t".join(text_parts).strip()
            if not text:
                continue

            effective_label, normalized = normalize_label(raw_label, last_label)
            if effective_label is None or normalized is None:
                continue

            last_label = effective_label
            yield text, normalized


def collect_files(input_dir: Path) -> List[Path]:
    """Return sorted list of .txt files under input_dir."""

    return sorted(p for p in input_dir.rglob("*.txt") if p.is_file())


def write_jsonl(records: Iterable[Tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for sentence, label in records:
            obj = {"sentence": sentence, "label": label}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def convert(input_dir: Path, output_path: Path) -> None:
    files = collect_files(input_dir)
    records: List[Tuple[str, str]] = []
    for file_path in files:
        records.extend(iter_annotations(file_path))
    write_jsonl(records, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Webis annotated editorials to JSONL for MCC.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Root of annotated-txt/split-by-portal-final")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path (default: src/data/editorials.jsonl)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.input, args.output)
    print(f"Wrote JSONL to {args.output}")


if __name__ == "__main__":
    main()
