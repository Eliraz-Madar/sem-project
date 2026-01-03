# sem-project

A two-stage ML pipeline for classifying news articles as **fact** or **opinion** using argument component analysis:
1. **Stage 1 (MCC)**: BERT-based sentence classifier for argument components (claim/premise/other)
2. **Stage 2 (Document)**: SVM and RNN classifiers using MCC-extracted features

## Quick Start

### Prerequisites
- Python 3.10+
- Dependencies: `pip install -r requirements.txt` (torch, transformers, scikit-learn, etc.)

### Training Pipeline

**Step 1: Prepare MCC Training Data**

The repository includes the Webis Editorials corpus (`src/corpus-webis-editorials-16/annotated-txt/`). Convert it to JSONL format:

```powershell
cd src
python -m scripts.convert_webis_to_jsonl
```

This creates `src/data/editorials.jsonl` (56,967 sentence-level examples with labels: `claim`, `premise`, `other`).

**Step 2: Train the MCC Sentence Classifier**

```powershell
cd src
python -m scripts.run_mcc_training --train data/mcc_train.jsonl --dev data/mcc_dev.jsonl
```

Default data files (`mcc_train.jsonl`, `mcc_dev.jsonl`) should already exist in `src/data/`. This saves the trained model to `src/checkpoints/`.

**Step 3: Train Document Classifiers**

```powershell
cd src
python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_webis_tiny.pt
```

This:
- Extracts features from documents using the trained MCC model
- Trains an SVM on argument component ratio features
- Trains an RNN on argument component sequences
- Reports macro-F1 scores for both models

## Data Files

**Essential datasets** (included):
- `src/data/editorials.jsonl` - Webis corpus (sentence-level, 56K+ examples)
- `src/data/mcc_train.jsonl`, `src/data/mcc_dev.jsonl` - MCC training/validation splits
- `src/data/news_train.jsonl`, `src/data/news_test.jsonl` - Document-level fact/opinion data
- `src/checkpoints/mcc_webis_tiny.pt` - Pre-trained MCC checkpoint

## Project Structure

```
src/
├── mcc/                    # Stage 1: Sentence-level argument classifier
│   ├── data.py            # JSONL reader, Dataset, DataLoader
│   ├── models.py          # BertMCCClassifier (BERT wrapper)
│   └── train.py           # Training loop, evaluation
├── doc_clf/               # Stage 2: Document-level fact/opinion classifier
│   ├── features.py        # Feature extraction using MCC
│   ├── models.py          # SVM wrapper, ArgumentRNNClassifier (GRU)
│   └── train.py           # Training logic for both models
├── scripts/               # CLI entry points
│   ├── convert_webis_to_jsonl.py    # Data preparation
│   ├── run_mcc_training.py          # Train MCC
│   └── run_doc_classification.py    # Train document classifiers
└── data/                  # Datasets and checkpoints
```

## Archived Files

External data collection scripts (NYT API, RSS scrapers, etc.) have been moved to `archive/` as they are not required for the core ML pipeline. See `archive/README.md` for details.

## Notes

- All scripts use UTF-8 encoding
- Training requires GPU for reasonable speed (falls back to CPU if unavailable)
- The pipeline is deterministic with fixed random seeds