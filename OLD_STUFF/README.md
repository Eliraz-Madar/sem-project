# Multi-Model Comparison Notebooks

This directory contains Jupyter notebooks for systematic comparison of multiple machine learning models on the Fact vs Opinion classification task.

## Overview

The project has **two stages**:
1. **Stage 1 (MCC)**: Sentence-level argument component classification (claim/premise/other)
2. **Stage 2 (Doc Classification)**: Document-level fact vs opinion classification

This folder now contains **both**:
- Full start-to-finish notebooks (sentence → doc classifier)
- Model-comparison notebooks (Stage 2 analysis)

## Notebooks

### Start-to-Finish (Run These in Order)

1) [mcc_data_prep.ipynb](mcc_data_prep.ipynb)
**Purpose**: Convert Webis annotations to MCC JSONL data.

2) [mcc_training.ipynb](mcc_training.ipynb)
**Purpose**: Train the MCC sentence classifier (Stage 1).

3) [doc_clf_training.ipynb](doc_clf_training.ipynb)
**Purpose**: Train document-level classifiers (SVM + RNN) using MCC features (Stage 2).

Optional:
- [trace_pipeline.ipynb](trace_pipeline.ipynb) — end-to-end trace and reporting.

### 1. [01_environment_and_data_check.ipynb](01_environment_and_data_check.ipynb)
**Purpose**: Verify environment setup and validate datasets.

**What it does:**
- Detects GPU/CPU and key package versions
- Validates repository structure
- Loads and inspects training/test data
- Shows label distribution and sample documents
- Performs final readiness check

**Run time**: ~1-2 minutes

**Output**: None (diagnostic only)

---

### 2. [02_doc_clf_model_comparison.ipynb](02_doc_clf_model_comparison.ipynb)
**Purpose**: Compare multiple transformer backbones for document-level classification.

**What it does:**
- Tests 3 models by default:
  - `prajjwal1/bert-tiny` (very small, quick baseline)
  - `bert-base-uncased` (standard BERT)
  - `distilbert-base-uncased` (distilled BERT)
- For each model:
  - Trains **SVM** on MCC ratio features
  - Trains **RNN** on MCC tag sequences
  - Computes metrics: macro-F1, weighted-F1, accuracy
  - Generates confusion matrices
- Creates results table sorted by test F1
- Saves plots and confusion matrices

**Key Features:**
- Fixed seed (SEED=42) for reproducibility
- Consistent preprocessing across all models
- Uses existing MCC checkpoint or trains a minimal one

**Run time**: 10-20 minutes (depending on device)

**Outputs** (saved to `outputs/experiments/`):
- `doc_clf_comparison_<timestamp>.csv` — Results table
- `confusion_matrix_<model>_<timestamp>.png` — Confusion matrices

**Example Output:**
```
SVM CLASSIFIER RESULTS
Model                    Test F1 (macro)  Test Accuracy
bert-base-uncased        0.8234          0.8156
distilbert-base-uncased  0.7921          0.7834
prajjwal1/bert-tiny      0.6543          0.6234
```

---

### 3. [03_embedding_baseline_comparison.ipynb](03_embedding_baseline_comparison.ipynb)
**Purpose**: Compare embedding-based approaches vs fine-tuned transformers.

**What it does:**
- **Embedding-based methods** (fast, simple):
  - SBERT (`all-MiniLM-L6-v2`) + Logistic Regression
  - SBERT + Linear SVM
- **Fine-tuned transformer** (more powerful):
  - BERT-base fine-tuned end-to-end
- Evaluates all approaches with same metrics
- Compares speed/complexity vs accuracy tradeoff

**Key Features:**
- Uses smaller SBERT model for speed
- Automatic package installation if needed
- Clear comparison of embedding vs transformer paradigms

**Run time**: 15-25 minutes

**Outputs** (saved to `outputs/experiments/`):
- `embedding_comparison_<timestamp>.csv` — Results table

**Example Output:**
```
EMBEDDING vs TRANSFORMER COMPARISON
Approach                          Type          Test F1 (macro)  Test Accuracy
SBERT + Linear SVM                embedding     0.7845          0.7756
SBERT + LogisticRegression        embedding     0.7623          0.7534
BERT-base Fine-tuned              transformer   0.8234          0.8156
```

---

## How to Run (From Start to Finish)

### Step 0: Environment Check (Recommended)
Run [01_environment_and_data_check.ipynb](01_environment_and_data_check.ipynb) to verify packages and data.

### Step 1: Sentence Data Prep (Stage 1 data)
Run [mcc_data_prep.ipynb](mcc_data_prep.ipynb).

### Step 2: Train Sentence Classifier (Stage 1)
Run [mcc_training.ipynb](mcc_training.ipynb).

### Step 3: Train Document Classifiers (Stage 2)
Run [doc_clf_training.ipynb](doc_clf_training.ipynb).

### Optional: Pipeline Trace/Report
Run [trace_pipeline.ipynb](trace_pipeline.ipynb) for a full trace with timing + report files.

### Run from VS Code
1. Open the notebook file
2. Select Python kernel
3. Click "Run All" or run cells individually

---

## Key Hyperparameters (Fixed for Reproducibility)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SEED` | 42 | Fixed for reproducibility |
| `DEVICE` | Auto-detect | GPU if available, else CPU |
| `batch_size` (SVM/RNN) | 4 | For doc classification |
| `batch_size` (Transformer) | 4 | For fine-tuning |
| `epochs` (RNN) | 2 | Training epochs |
| `epochs` (Transformer) | 2 | Fine-tuning epochs |
| `max_length` | 512 (BERT) | Token limit for transformers |
| `learning_rate` (RNN) | 1e-3 | Adam optimizer |
| `learning_rate` (BERT fine-tune) | 2e-5 | AdamW optimizer |

---

## Data Files Required

Ensure these files exist in `src/data/`:
- `mcc_train.jsonl` — MCC training sentences
- `mcc_dev.jsonl` — MCC validation sentences
- `news_train.jsonl` — Document training data
- `news_test.jsonl` — Document test data

If missing, run the data preparation scripts:
```bash
cd src
python -m scripts.convert_webis_to_jsonl
```

---

## MCC Checkpoint

Notebook 2 requires an MCC checkpoint at `src/checkpoints/mcc_webis_tiny.pt`. If missing:
1. The notebook will create a minimal checkpoint automatically, OR
2. You can manually train one:
```bash
cd src
python -m mcc.train --epochs 2 --batch-size 8
```

---

## Results Storage

All results are saved to `outputs/experiments/` with timestamp:
```
outputs/experiments/
├── doc_clf_comparison_20240119_120530.csv
├── embedding_comparison_20240119_121045.csv
├── confusion_matrix_bert-base-uncased_20240119_120530.png
├── confusion_matrix_distilbert-base-uncased_20240119_120530.png
└── ...
```

---

## Metrics Explained

### Macro-F1 (Primary Metric)
- Unweighted average of F1 for each class
- Good for imbalanced datasets
- Penalizes poor performance on minority class

### Weighted-F1
- Class-weighted average of F1
- Accounts for class frequency
- Closer to accuracy for imbalanced data

### Accuracy
- Percentage of correct predictions
- Can be misleading for imbalanced data

### Confusion Matrix
```
         Predicted
Actual   Fact  Opinion
Fact     TN    FP
Opinion  FN    TP
```

---

## Troubleshooting

### ImportError: sentence_transformers
```python
pip install sentence-transformers
```

### CUDA out of memory
- Reduce `batch_size` in notebooks
- Use `prajjwal1/bert-tiny` instead of larger models
- Disable GPU: set `DEVICE = torch.device("cpu")`

### Checkpoint not found
The notebooks auto-create minimal checkpoints if missing.

### Slow execution
- Use GPU (NVIDIA CUDA)
- Use distilled models (DistilBERT, SBERT MiniLM)
- Reduce epochs or batch size

---

## Extending the Notebooks

### Add a New Model
In notebook 2, modify `MODELS_TO_TEST`:
```python
MODELS_TO_TEST = [
    "prajjwal1/bert-tiny",
    "bert-base-uncased",
    "roberta-base",  # Add new model
    "distilbert-base-uncased",
]
```

### Add a New Baseline Method
In notebook 3, add after line "Logistic Regression":
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_model.fit(train_embeddings, train_labels)
rf_test_preds = rf_model.predict(test_embeddings)
embedding_results["SBERT + Random Forest"] = { ... }
```

### Save Custom Results
```python
# Save to custom path
export_df.to_csv(REPO_ROOT / "my_results.csv", index=False)
```

---

## Performance Baselines

Expected performance on fact/opinion dataset:

| Approach | F1 (macro) | Speed | Complexity |
|----------|-----------|-------|-----------|
| SBERT + LogReg | ~0.76 | ⚡⚡⚡ | ⭐ |
| SBERT + SVM | ~0.78 | ⚡⚡ | ⭐⭐ |
| BERT-base Fine-tune | ~0.82 | ⚡ | ⭐⭐⭐ |

*Note: Actual results depend on dataset size and quality.*

---

## References

- **BERT**: Devlin et al., 2019. "BERT: Pre-training of Deep Bidirectional Transformers"
- **SBERT**: Reimers & Gurevych, 2019. "Sentence-BERT"
- **DistilBERT**: Sanh et al., 2019. "DistilBERT, a distilled version of BERT"

---

## Contact & Issues

For questions or issues with the notebooks:
1. Check the Troubleshooting section above
2. Review cell output for error messages
3. Verify data files exist in `src/data/`
4. Check repository README for setup instructions
