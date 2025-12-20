# Data Acquisition Audit

This file summarizes scripts and code in the repository that perform scraping, downloading, or data conversion for document-level datasets.

Findings
-------

- `src/scripts/scrape_news_to_jsonl.py`
  - Summary: Scrapes factual vs opinion news articles by fetching RSS feeds, following links, downloading pages, and extracting article body text using `requests` + `beautifulsoup4`.
  - Saves to: `data/news_train.jsonl` (default).
  - How to run: from repository root or `src/`:
    ```bash
    python -m scripts.scrape_news_to_jsonl --max-per-label 50
    ```
  - Notes: Requires `requests` and `beautifulsoup4`; polite defaults (sleep, limits) are configured.

- `src/scripts/convert_webis_to_jsonl.py`
  - Summary: Converts the included Webis editorials corpus (sentence-level annotations) into sentence-level MCC JSONL format. This is a conversion script (not a web scraper).
  - Saves to: `src/data/editorials.jsonl` (default).
  - How to run: from repo root or `src/`:
    ```bash
    python -m scripts.convert_webis_to_jsonl
    ```

- `src/scripts/quick_smoke.py`
  - Summary: Writes a tiny smoke dataset under `tmp_smoke/data` for quick local testing (small set of docs / MCC train/dev samples).
  - Saves to: `src/tmp_smoke/data/docs.jsonl`, `mcc_train.jsonl`, etc.
  - How to run: from `src/`:
    ```bash
    python -m scripts.quick_smoke
    ```

No other web-scraping libraries (selenium, playwright, scrapy, newspaper3k, gdown, kaggle API, etc.) were found in the `src/` codebase.

Recommendations
---------------

- Use `src/scripts/scrape_news_to_jsonl.py` if you want to fetch live factual/opinion articles (internet required). It produces the `data/news_train.jsonl` format expected by the document-level pipeline.
- For offline/demo testing, prefer generating a deterministic synthetic dataset (a small script is provided as `src/scripts/prepare_news_dataset.py`).
