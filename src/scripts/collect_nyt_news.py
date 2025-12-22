"""
Collect NYT articles via Archive API and save labeled CSV.

Usage:
- Set environment variables: NYT_API_KEY, YEAR, MONTH
- Or pass CLI args: --api-key, --year, --month

Output:
- Creates 'nyt_news_collection.csv' in the current working directory
  with columns: text,label
"""

import os
import time
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException

try:
    from newspaper import Article
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "newspaper3k is required. Please install via 'pip install newspaper3k'."
    ) from e

NYT_API_KEY = "N8oNz5nek4zihXPTiMAlAMsxSOmlZEGOIXjnRIarAZf9LbGg"
YEAR = 2023
MONTH = 5
NYT_ARCHIVE_URL_TEMPLATE = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
OUTPUT_CSV = "nyt_news_collection.csv"

# Mapping for type_of_material to our labels
OPINION_TYPES = {"Op-Ed", "Editorial", "Letter"}
FACT_TYPES = {"News"}


def get_config_from_env_or_args() -> Tuple[str, int, int]:
    parser = argparse.ArgumentParser(description="Collect NYT articles for a given month")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("NYT_API_KEY"))
    parser.add_argument("--year", dest="year", type=int, default=os.getenv("YEAR"))
    parser.add_argument("--month", dest="month", type=int, default=os.getenv("MONTH"))
    args = parser.parse_args([] if os.getenv("PYTEST_CURRENT_TEST") else None)

    if not args.api_key:
        raise ValueError("NYT_API_KEY is required (env var or --api-key)")
    if args.year is None:
        raise ValueError("YEAR is required (env var or --year)")
    if args.month is None:
        raise ValueError("MONTH is required (env var or --month)")

    # Basic bounds check
    if not (1851 <= int(args.year) <= 2100):
        raise ValueError("YEAR must be between 1851 and 2100")
    if not (1 <= int(args.month) <= 12):
        raise ValueError("MONTH must be between 1 and 12")

    return str(args.api_key), int(args.year), int(args.month)


def fetch_archive_docs(api_key: str, year: int, month: int) -> List[Dict]:
    url = NYT_ARCHIVE_URL_TEMPLATE.format(year=year, month=month)
    params = {"api-key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except RequestException as e:
        raise RuntimeError(f"Failed to fetch NYT archive: {e}")

    data = resp.json()
    # Expected structure: { 'response': { 'docs': [...] } }
    response = data.get("response", {})
    docs = response.get("docs", [])
    if not isinstance(docs, list):
        raise RuntimeError("Unexpected response format: 'docs' is not a list")
    return docs


def map_type_to_label(type_of_material: Optional[str]) -> Optional[str]:
    if not type_of_material:
        return None
    # Exact match per requirements
    if type_of_material in OPINION_TYPES:
        return "opinion"
    if type_of_material in FACT_TYPES:
        return "fact"
    return None


def extract_article_text(url: str, sleep_seconds: float = 1.0) -> Optional[str]:
    # Respect rate limits with a small delay before each download
    time.sleep(sleep_seconds)
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        if len(text) < 200:  # Skip extremely short/empty parses
            return None
        return text
    except Exception:
        return None


def collect_nyt_news() -> None:
    api_key, year, month = get_config_from_env_or_args()
    print(f"Fetching NYT archive for {year}-{month:02d}...")
    docs = fetch_archive_docs(api_key, year, month)
    print(f"Fetched {len(docs)} documents from archive. Filtering...")

    rows: List[Tuple[str, str]] = []
    for d in docs:
        type_of_material = d.get("type_of_material")
        web_url = d.get("web_url")
        label = map_type_to_label(type_of_material)
        if not label or not web_url:
            continue

        text = extract_article_text(web_url, sleep_seconds=1.0)
        if not text:
            continue

        rows.append((text, label))

    print(f"Collected {len(rows)} labeled articles. Writing CSV...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, label])

    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "main" or __name__ == "__main__":
    collect_nyt_news()
