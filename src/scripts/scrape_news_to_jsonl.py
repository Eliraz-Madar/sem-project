"""Scrape a small factual vs opinion dataset into data/news_train.jsonl.

Educational/research use only. Polite defaults: small limits and sleep between requests.
The script fetches RSS/section feeds, grabs article URLs, downloads pages, extracts
main text, and writes JSONL with fields {"text", "label", "title", "url"}.

Usage (from repo root or src/):
    python -m scripts.scrape_news_to_jsonl --max-per-label 50

Requirements: requests, beautifulsoup4, tqdm
"""
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure your sources (public RSS feeds or section URLs)
FACTUAL_SOURCES = [
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
]

OPINION_SOURCES = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Opinion.xml",
    "https://www.theguardian.com/uk/commentisfree/rss",
    "https://www.washingtonpost.com/opinions/rss",
]

DEFAULT_OUTPUT = Path("data/news_train.jsonl")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_rss_entries(url: str) -> List[str]:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []
    links: List[str] = []
    for item in root.findall(".//item"):
        link_el = item.find("link")
        if link_el is not None and link_el.text:
            links.append(link_el.text.strip())
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        for link_el in entry.findall("{http://www.w3.org/2005/Atom}link"):
            href = link_el.attrib.get("href")
            if href:
                links.append(href.strip())
    return links


def clean_paragraphs(paragraphs: List[str], min_len: int = 30) -> str:
    cleaned = [p.strip() for p in paragraphs if p and len(p.strip()) >= min_len]
    return "\n\n".join(cleaned).strip()


def extract_article_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    article_tag = soup.find("article")
    if article_tag:
        candidates.append(article_tag)
    for cls_key in (
        "article-body",
        "articleBody",
        "story-body",
        "post-content",
        "content__article-body",
        "entry-content",
    ):
        div = soup.find("div", class_=lambda c: c and cls_key in c)
        if div:
            candidates.append(div)
    if not candidates and soup.body:
        candidates.append(soup.body)

    for cand in candidates:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in cand.find_all("p")]
        text = clean_paragraphs(paragraphs)
        if text:
            return text

    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = clean_paragraphs(paragraphs)
    return text or None


def fetch_article(url: str, sleep_sec: float, min_length: int) -> Optional[dict]:
    time.sleep(max(0.0, sleep_sec))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; edu-research-bot/0.1)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    text = extract_article_text(resp.text)
    if not text or len(text) < min_length:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else None
    return {"text": text, "title": title, "url": url}


def scrape_category(
    feeds_or_urls: List[str],
    label: str,
    max_per_label: int,
    min_length: int,
    sleep_sec: float,
) -> List[dict]:
    collected: List[dict] = []
    seen: Set[str] = set()

    for source in feeds_or_urls:
        if len(collected) >= max_per_label:
            break
        links = fetch_rss_entries(source)
        for link in tqdm(links, desc=f"{label}-links", leave=False):
            if len(collected) >= max_per_label:
                break
            if link in seen:
                continue
            seen.add(link)
            article = fetch_article(link, sleep_sec=sleep_sec, min_length=min_length)
            if article:
                article["label"] = label
                collected.append(article)
    return collected


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape factual/opinion news into JSONL for training.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path (default: data/news_train.jsonl)")
    parser.add_argument("--max-per-label", type=int, default=50, help="Max articles to fetch per label")
    parser.add_argument("--min-length", type=int, default=300, help="Minimum article body length")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Collecting factual articles...")
    factual = scrape_category(
        FACTUAL_SOURCES,
        label="factual",
        max_per_label=args.max_per_label,
        min_length=args.min_length,
        sleep_sec=args.sleep,
    )

    print("Collecting opinion articles...")
    opinion = scrape_category(
        OPINION_SOURCES,
        label="opinion",
        max_per_label=args.max_per_label,
        min_length=args.min_length,
        sleep_sec=args.sleep,
    )

    all_rows = factual + opinion
    if not all_rows:
        print("No articles collected. Check sources or connectivity.")
        return

    write_jsonl(args.output, all_rows)
    print(f"Wrote {len(factual)} factual and {len(opinion)} opinion articles to {args.output}")


if __name__ == "__main__":
    main()
