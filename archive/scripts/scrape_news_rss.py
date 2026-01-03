"""Fast RSS feed scraper for news articles with fact/opinion classification.

This is the fastest approach since RSS feeds are structured (no HTML parsing needed).
Classifies articles as 'fact' or 'opinion' based on source category.

Usage:
    From src/: python -m scripts.scrape_news_rss --output data/news_articles.csv --max-articles 100

Optional args:
    --output: CSV file path (default: data/news_articles.csv)
    --max-articles: Max articles to fetch (default: 100)
    --timeout: Request timeout in seconds (default: 10)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import requests
import feedparser
from bs4 import BeautifulSoup

# RSS feeds: (url, category, is_opinion)
FEEDS = [
    # Hard news sources (fact)
    ("https://feeds.theguardian.com/theguardian/world/rss", "news", False),
    ("https://feeds.theguardian.com/theguardian/international/rss", "news", False),
    ("https://feeds.theguardian.com/theguardian/us-news/rss", "news", False),
    ("http://rss.cnn.com/rss/edition.rss", "news", False),
    ("http://rss.cnn.com/rss/cnn_us.rss", "news", False),
    ("https://feeds.npr.org/1001/rss.xml", "news", False),
    ("https://feeds.npr.org/1002/rss.xml", "news", False),
    ("https://feeds.bloomberg.com/markets/news.rss", "news", False),
    ("https://feeds.bloomberg.com/news/news.rss", "news", False),
    ("http://feeds.washingtonpost.com/rss/world", "news", False),
    ("http://feeds.washingtonpost.com/rss/national", "news", False),
    ("https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "news", False),
    ("https://rss.nytimes.com/services/xml/rss/nyt/US.xml", "news", False),
    
    # Opinion/Analysis sources
    ("https://feeds.theguardian.com/theguardian/commentisfree/rss", "opinion", True),
    ("https://feeds.theguardian.com/theguardian/commentisfree/debate/rss", "opinion", True),
    ("http://feeds.washingtonpost.com/rss/opinions", "opinion", True),
    ("https://rss.nytimes.com/services/xml/rss/nyt/Opinion.xml", "opinion", True),
    ("https://www.huffpost.com/section/opinion/feed", "opinion", True),
    ("https://www.vox.com/rss/index.xml", "opinion", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape news from RSS feeds and save to CSV")
    parser.add_argument("--output", type=Path, default=Path("data/news_articles.csv"))
    parser.add_argument("--max-articles", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--balance", action="store_true", help="Balance fact/opinion ratio")
    return parser.parse_args()


def extract_full_article(url: str, timeout: int = 5) -> str | None:
    """Try to extract full article text from URL using BeautifulSoup.
    Returns article text or None if fetch fails or times out.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Try common article selectors
        article_selectors = [
            "article", "main", "div.article-body", "div.article-content",
            "div.story-body", "div.story-body-text", "div[data-testid='storyBody']",
            "div.entry-content", "div.post-content", "div.content-body"
        ]
        
        article_elem = None
        for selector in article_selectors:
            elem = soup.select_one(selector)
            if elem:
                article_elem = elem
                break
        
        if not article_elem:
            # Fallback: get body text
            article_elem = soup.body or soup
        
        # Extract text and clean
        text = article_elem.get_text(separator=" ", strip=True)
        text = " ".join(text.split())  # Remove extra whitespace
        
        # Return if substantial length (at least 300 chars of actual content)
        if len(text) > 300:
            return text
        return None
    except requests.Timeout:
        return None
    except Exception:  # noqa: BLE001
        return None


def fetch_feed(feed_url: str, category: str, is_opinion: bool, timeout: int = 10) -> List[Dict]:
    """Fetch articles from a single RSS feed."""
    articles = []
    try:
        print(f"  Fetching {feed_url[:50]}... ({category})", end=" ")
        response = requests.get(feed_url, timeout=timeout)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        
        for entry in feed.entries[:30]:  # Fetch more per feed
            title = entry.get("title", "").strip()
            # Try multiple fields for body text
            text = entry.get("summary", "") or entry.get("description", "") or entry.get("content", "")
            if isinstance(text, list):  # Some feeds have content as list
                text = text[0].get("value", "") if text else ""
            text = text.strip()
            
            # If summary is short, try to fetch full article from URL
            if len(text) < 300:
                url = entry.get("link", "")
                if url:
                    full_text = extract_full_article(url, timeout=5)
                    if full_text:
                        text = full_text
            
            # Skip very short articles
            if len(text) < 100:
                continue
            
            # Determine label: use feed classification, fallback to keywords if needed
            label = "opinion" if is_opinion else "fact"
            
            # Keyword-based detection for additional opinion articles in fact feeds
            opinion_keywords = ["i believe", "i think", "in my view", "my opinion", "opinion",
                              "analysis", "argument", "perspective", "commentary", "editorial"]
            text_lower = text.lower()
            if not is_opinion and any(keyword in text_lower for keyword in opinion_keywords):
                label = "opinion"
            articles.append({
                "title": title,
                "full_text": text,
                "category": label,
                "source": category,
            })
        
        print(f"✓ ({len(articles)} articles)")
        return articles
    except requests.RequestException as e:
        print(f"✗ (error: {e})")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"✗ (parse error: {e})")
        return []


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Scraping RSS feeds (max {args.max_articles} articles)...")
    all_articles: List[Dict] = []
    
    for feed_url, category, is_opinion in FEEDS:
        if len(all_articles) >= args.max_articles:
            break
        articles = fetch_feed(feed_url, category, is_opinion, args.timeout)
        all_articles.extend(articles)
    
    # Trim to max
    all_articles = all_articles[: args.max_articles]
    
    # Balance if requested
    if args.balance and all_articles:
        fact_articles = [a for a in all_articles if a["category"] == "fact"]
        opinion_articles = [a for a in all_articles if a["category"] == "opinion"]
        
        # Aim for 50/50 split
        target = len(all_articles) // 2
        if len(fact_articles) > target:
            fact_articles = fact_articles[:target]
        if len(opinion_articles) > target:
            opinion_articles = opinion_articles[:target]
        
        all_articles = fact_articles + opinion_articles
    
    # Write CSV
    if not all_articles:
        print("No articles fetched. Check feed URLs and internet connection.")
        return
    
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "full_text", "category", "source"])
        writer.writeheader()
        writer.writerows(all_articles)
    
    # Summary
    fact_count = sum(1 for a in all_articles if a["category"] == "fact")
    opinion_count = sum(1 for a in all_articles if a["category"] == "opinion")
    
    print(f"\n✓ Saved {len(all_articles)} articles to {args.output}")
    print(f"  - Fact-based: {fact_count}")
    print(f"  - Opinion: {opinion_count}")
    print(f"  - Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
