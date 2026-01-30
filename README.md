# sem-project

Two-stage ML pipeline for classifying news articles as **fact** or **opinion** using argument component analysis:
1. **Stage 1 (MCC)**: BERT-based sentence classifier for argument components (claim/premise/other)
2. **Stage 2 (Document)**: SVM and RNN classifiers using MCC-extracted features

## Quick Start (Recommended: Notebooks)

### Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

### Run in Notebooks (Start → Finish)
Open these notebooks in order and run cells top-to-bottom:
1. [notebooks/mcc_data_prep.ipynb](notebooks/mcc_data_prep.ipynb) — builds MCC JSONL (skip if `src/data/editorials.jsonl` already exists)
2. [notebooks/mcc_training.ipynb](notebooks/mcc_training.ipynb) — trains the MCC sentence model
3. [notebooks/doc_clf_training.ipynb](notebooks/doc_clf_training.ipynb) — trains SVM + RNN document classifiers

Optional:
- [notebooks/trace_pipeline.ipynb](notebooks/trace_pipeline.ipynb) — full trace + report
- [notebooks/02_doc_clf_model_comparison.ipynb](notebooks/02_doc_clf_model_comparison.ipynb) — model comparisons
- [notebooks/03_embedding_baseline_comparison.ipynb](notebooks/03_embedding_baseline_comparison.ipynb) — embedding baselines

## Quick Start (CLI)

### Step 1: Prepare MCC Training Data
```powershell
cd src
python -m scripts.convert_webis_to_jsonl
```
Creates `src/data/editorials.jsonl` (sentence-level examples).

### Step 2: Train the MCC Sentence Classifier
```powershell
cd src
python -m scripts.run_mcc_training --train data/mcc_train.jsonl --dev data/mcc_dev.jsonl
```
Saves a checkpoint to `src/checkpoints/`.

### Step 3: Train Document Classifiers
```powershell
cd src
python -m scripts.run_doc_classification --news-train data/news_train.jsonl --news-test data/news_test.jsonl --mcc-checkpoint checkpoints/mcc_webis_tiny.pt
```
Trains SVM + RNN and reports macro-F1 scores.

## Data Files

Required files in `src/data/`:
- `mcc_train.jsonl`, `mcc_dev.jsonl`
- `news_train.jsonl`, `news_test.jsonl`

Optional (only if you want to rebuild MCC data):
- `editorials.jsonl` (produced by the data prep step)

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
└── data/                  # Datasets
checkpoints/               # Trained models
```

## Notes

- All scripts use UTF-8 encoding
- Training requires GPU for reasonable speed (falls back to CPU if unavailable)
- The pipeline is deterministic with fixed random seeds