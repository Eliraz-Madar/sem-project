# Automation: run_pipeline_no_mcc.bat

This document explains how to run the one-click pipeline runner that starts after MCC training.

What the runner does
- Creates and activates a local virtual environment at `src/.venv` (if missing).
- Installs dependencies (from `requirements.txt` at repo root if present, otherwise a minimal set).
- Ensures `src/data/news_train.jsonl` and `src/data/news_test.jsonl` exist (generates deterministic demo data if missing).
- Verifies the MCC checkpoint exists at `src/checkpoints/mcc_bert.pt` (will error if missing).
- Runs the document-level classification script:
  `python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_bert.pt`

How to run (Windows)
1. Open cmd.exe or PowerShell.
2. From the repository root (where `run_pipeline_no_mcc.bat` is located), run:

```powershell
.\run_pipeline_no_mcc.bat
```

Notes
- The runner assumes MCC training has already been completed and a checkpoint exists at `src/checkpoints/mcc_bert.pt`.
- If you prefer to use your own dataset, place `news_train.jsonl` and `news_test.jsonl` under `src/data/` (JSONL lines must include `text` and `label`).
- The script `src/scripts/prepare_news_dataset.py` generates deterministic synthetic data for offline/demo testing.

Running from `src/` directly
- You can run the pipeline steps manually from the `src/` directory as well:

```powershell
cd src
python -m scripts.prepare_news_dataset
python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_bert.pt
```
