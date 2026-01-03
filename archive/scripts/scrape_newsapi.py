"""Scrape news from NewsAPI.org (requires free API key).

Get free API key at: https://newsapi.org/register

Usage:
    From src/: python -m scripts.scrape_newsapi --api-key YOUR_KEY --output data/newsapi_articles.csv --max-articles 500
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape news from NewsAPI.org")
    parser.add_argument("--api-key", type=str, required=True, help="NewsAPI.org API key (get free key at https://newsapi.org)")
    parser.add_argument("--output", type=Path, default=Path("data/newsapi_articles.csv"))
    parser.add_argument("--max-articles", type=int, default=500)
    parser.add_argument("--country", type=str, default="us", help="Country code (us, gb, in, etc.)")
    return parser.parse_args()


def fetch_top_headlines(api_key: str, country: str, max_articles: int) -> List[Dict]:
    """Fetch top headlines from NewsAPI."""
    articles = []
    url = "https://api.nytimes.com/svc/topstories/v2"
    
    # Use New York Times API (free, no key needed for basic usage)
    categories = ["world", "us", "politics", "business", "opinion"]
    
    for category in categories:
        if len(articles) >= max_articles:
            break
        
        try:
            # Try NewsAPI first
            resp = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": country, "pageSize": 100, "apiKey": api_key},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                for article in data.get("articles", []):
                    if len(articles) >= max_articles:
                        break
                    
                    title = article.get("title", "").strip()
                    text = article.get("description", "").strip()
                    
                    if len(text) > 50:
                        articles.append({
                            "title": title,
                            "full_text": text,
                            "category": "news",
                            "source": article.get("source", {}).get("name", "Unknown"),
                        })
        except Exception as e:
            print(f"Error fetching {category}: {e}")
            continue
    
    return articles


def fetch_nyt_api(max_articles: int = 500) -> List[Dict]:
    """Fetch from New York Times API (free, no key needed for basic usage)."""
    articles = []
    base_url = "https://api.nytimes.com/svc/topstories/v2"
    
    # Note: NYT requires API key. Use NewsAPI instead for now.
    return articles


def main() -> None:
    args = parse_args()
    
    print(f"Fetching articles from NewsAPI.org...")
    articles = fetch_top_headlines(args.api_key, args.country, args.max_articles)
    
    if not articles:
        print("No articles fetched. Check your API key and internet connection.")
        print("Get a free key at: https://newsapi.org/register")
        raise SystemExit(1)
    
    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "full_text", "category", "source"])
        writer.writeheader()
        writer.writerows(articles)
    
    print(f"✓ Saved {len(articles)} articles to {args.output}")


if __name__ == "__main__":
    main()
