# Archive Directory

This directory contains files that are **not required** for the core ML training pipeline but may be useful for reference or future data collection efforts.

## Contents

### `docs/`
- `automation.md` - Documentation about external data collection scripts (NYT API, RSS feeds, NewsAPI)

### `scripts/`
Data collection and conversion utilities:
- `collect_nyt_news.py` - NYT Archive API scraper
- `scrape_news_rss.py` - RSS feed scraper for news sources
- `scrape_newsapi.py` - NewsAPI.org integration
- `convert_csv_to_jsonl.py` - Generic CSV to JSONL converter
- `import_fact_opinion_csvs.py` - Merges fact/opinion CSV files
- `temp.py` - Experimental Google News scraper
- `prepare_news_dataset.py` - Generates synthetic test data

### `data/`
- `news_articles.csv` - Scraped RSS articles (189 rows, source for news_train/test)

### `corpus-webis/`
Unused Webis corpus formats:
- `unannotated.csv` - Raw editorials without annotations
- `annotated-xmi/` - UIMA XMI format annotations (not used by Python scripts)
- `uima-type-systems/` - UIMA XML type definitions

### Root
- `nyt_news_collection.csv` - Sample NYT articles from API (10 examples)

## Why These Files Were Archived

These files were moved out of the main source tree because:
1. They are **not used** by the core ML training pipeline (MCC + document classification)
2. They involve **external data sources** that require API keys or web scraping
3. The current training workflow relies on the Webis corpus and pre-generated news datasets

## If You Need to Use Them

To restore any archived file back to the source tree:
```powershell
# Example: restore NYT scraper
git mv archive\scripts\collect_nyt_news.py src\scripts\collect_nyt_news.py
```

**Note:** Restored scripts may require additional dependencies or API keys not listed in the main `requirements.txt`.
