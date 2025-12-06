"""Quick smoke test for convert_webis_to_jsonl.

Runs the converter with default input/output, reports how many records were
written, and prints the first few lines to verify format.

Usage (from repo root or src/):
    python -m scripts.test
"""
from __future__ import annotations

from pathlib import Path

from scripts.convert_webis_to_jsonl import DEFAULT_INPUT, DEFAULT_OUTPUT, convert


def main() -> None:
    convert(DEFAULT_INPUT, DEFAULT_OUTPUT)

    total = 0
    lines: list[str] = []
    with DEFAULT_OUTPUT.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            total += 1
            if idx < 3:
                lines.append(line.rstrip("\n"))

    print(f"Wrote {total} records to {DEFAULT_OUTPUT}")
    if lines:
        print("Sample lines:")
        for item in lines:
            print(item)


if __name__ == "__main__":
    main()
