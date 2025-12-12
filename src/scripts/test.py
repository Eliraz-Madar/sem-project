"""Quick smoke test for convert_webis_to_jsonl.

Runs the converter with default input/output, reports how many records were
written, and prints the first few lines to verify format.

Usage (from repo root or src/):
    python -m scripts.test
"""
from __future__ import annotations

from pathlib import Path
import json

from scripts.convert_webis_to_jsonl import DEFAULT_INPUT, DEFAULT_OUTPUT, convert


def main() -> None:
    convert(DEFAULT_INPUT, DEFAULT_OUTPUT)
    # Write the first 10k JSONL rows to CSV (sentence,label)
    csv_path = DEFAULT_OUTPUT.with_suffix(".csv")
    total = 0
    samples: list[str] = []
    with DEFAULT_OUTPUT.open("r", encoding="utf-8") as fh, csv_path.open("w", encoding="utf-8") as csv_file:
        csv_file.write("sentence,label\n")
        for idx, line in enumerate(fh):
            total += 1
            if idx < 9999:
                samples.append(line.rstrip("\n"))
            if idx < 10000:
                try:
                    obj = json.loads(line)
                    sentence = obj.get("sentence", "").replace('"', "''")
                    label = obj.get("label", "").replace('"', "''")
                    csv_file.write(f'"{sentence}","{label}"\n')
                except json.JSONDecodeError:
                    continue

    print(f"Wrote {total} records to {DEFAULT_OUTPUT}")
    if samples:
        print("Sample lines:")
        for item in samples:
            print(item)
    print(f"\nWrote first 10000 records (or fewer if file shorter) to {csv_path}")


if __name__ == "__main__":
    main()
