# Quick Start Guide - Multi-Model Comparison Notebooks

## 🎯 What's New

Three Jupyter notebooks have been added to systematically compare multiple ML models for the Fact vs Opinion classification task. These notebooks add **zero complexity** to your existing training pipeline — they only analyze and compare.

---

## 📁 New Files

```
notebooks/                                   ← New folder
├── 01_environment_and_data_check.ipynb     ← Verify setup
├── 02_doc_clf_model_comparison.ipynb       ← Main comparison (BERT, DistilBERT, etc.)
├── 03_embedding_baseline_comparison.ipynb  ← SBERT vs fine-tuned BERT
└── (see top-level README.md)               ← Full documentation

outputs/
└── experiments/                             ← Results saved here
    ├── doc_clf_comparison_<timestamp>.csv
    ├── confusion_matrix_<model>_<timestamp>.png
    └── embedding_comparison_<timestamp>.csv

(Optional) trace pipeline guide: src/scripts/TRACE_PIPELINE_GUIDE.md
```

---

## 🚀 Quick Start

### Option 1: VS Code (Easiest)
```
1. Open: notebooks/01_environment_and_data_check.ipynb
2. Select Python kernel (bottom right)
3. Click "Run All"
4. Repeat for notebooks 02 and 03
```

### Option 2: Jupyter Lab
```bash
cd <PATH_TO_REPO_ROOT>
jupyter notebook
# Opens browser, click notebooks/01_...ipynb
```

### Option 3: Command Line
```bash
cd <PATH_TO_REPO_ROOT>/notebooks
jupyter nbconvert --to notebook --execute 01_environment_and_data_check.ipynb
```

---

## ✅ What Each Notebook Does

### Notebook 1: Environment Check (1-2 min)
- ✓ Detects GPU/CPU, checks Python packages
- ✓ Validates data files exist
- ✓ Shows dataset statistics
- 📤 **Output**: Diagnostic report (printed)

**Run this first** to verify everything is set up correctly.

---

### Notebook 2: Model Comparison (10-20 min) ⭐
- Compares 3 transformer models:
  - `prajjwal1/bert-tiny` (very fast)
  - `bert-base-uncased` (standard)
  - `distilbert-base-uncased` (compact)
- For each model:
  - ✓ SVM classification results
  - ✓ RNN classification results
  - ✓ Confusion matrices
  - ✓ F1 scores (macro & weighted)
- 📤 **Output**: 
  - `doc_clf_comparison_<timestamp>.csv` — Results table
  - `confusion_matrix_*.png` — Confusion matrices

**This is the main comparison notebook.**

---

### Notebook 3: Embedding Baseline (15-25 min)
- Compares two paradigms:
  - **Embedding-based** (SBERT + simple classifier)
  - **Fine-tuned transformer** (BERT end-to-end)
- Shows speed vs accuracy tradeoff
- 📤 **Output**: `embedding_comparison_<timestamp>.csv`

---

## 📊 Example Results

After running notebook 2, you'll see a table like:

```
Model                    SVM Test F1  RNN Test F1  Best for
bert-base-uncased        0.8234       0.7945       ← Overall best
distilbert-base-uncased  0.7921       0.7823
prajjwal1/bert-tiny      0.6543       0.6234       ← Baseline
```

Plus confusion matrices showing exactly what each model gets wrong.

---

## 🔑 Key Features

✅ **Reproducible**
- Fixed seed (SEED=42)
- Same preprocessing for all models
- Results are consistent across runs

✅ **Non-invasive**
- Zero changes to your existing code
- Notebooks import and use existing modules
- All original scripts still work unchanged

✅ **Ready to Run**
- Auto-creates missing directories
- Creates MCC checkpoint if needed
- Clear error messages if data missing

✅ **Production-ready Results**
- CSV files for importing to Excel/Python
- PNG plots for presentations
- Timestamped results tracking

---

## ⚠️ Troubleshooting

**"ModuleNotFoundError: No module named 'sentence_transformers'"**
→ Notebook 3 auto-installs it. If manual install needed:
```bash
pip install sentence-transformers
```

**"CUDA out of memory"**
→ Reduce batch_size in notebook (change `batch_size=4` to `batch_size=2`)
→ Use CPU instead (auto-detects, but slower)

**"FileNotFoundError: news_train.jsonl"**
→ Data files missing. Run setup first:
```bash
cd src
python -m scripts.convert_webis_to_jsonl
```

**"MCC checkpoint not found"**
→ Notebook auto-creates one. If issues, manually:
```bash
cd src
python -m mcc.train --epochs 1
```

For troubleshooting and project structure, see the top-level `README.md`.

---

## 📖 Full Documentation

- **Detailed guide**: See `notebooks/README.md`
- **Audit findings**: See `AUDIT_AND_IMPLEMENTATION_REPORT.md`

---

## 💡 What's Next?

### Add More Models
Edit notebook 2 line ~80:
```python
MODELS_TO_TEST = [
    "prajjwal1/bert-tiny",
    "bert-base-uncased",
    "roberta-base",           # ← Add your model here
    "distilbert-base-uncased",
]
```

### Run Custom Experiment
Create a new cell:
```python
# Test your specific model
result = run_experiment("your-model-name", train_labels, test_labels)
```

### Export Results
```python
export_df.to_excel("my_comparison.xlsx")  # Export to Excel
```

---

## 🎓 Key Takeaways

The notebooks implement:

1. **Systematic Comparison** — All models tested with identical preprocessing
2. **Comprehensive Metrics** — Macro-F1, weighted-F1, accuracy, confusion matrices
3. **Reproducibility** — Fixed seeds, logged hyperparameters
4. **Production Ready** — CSV/PNG outputs, clear best model identification
5. **Extensible** — Easy to add more models or baselines

---

## ✨ Summary

| Metric | Value |
|--------|-------|
| Notebooks | 3 new |
| Models compared | 3-4 |
| Baseline methods | 3 (SBERT+LR, SBERT+SVM, BERT fine-tune) |
| Metrics per model | 6+ (F1 macro/weighted, accuracy, confusion matrix) |
| Time to run all | ~30-50 minutes (GPU) |
| Code changes needed | ZERO |
| Results export | CSV + PNG |

---

**Start with**: `notebooks/01_environment_and_data_check.ipynb`

**Main comparison**: `notebooks/02_doc_clf_model_comparison.ipynb`

**Questions?** See `notebooks/README.md` for detailed docs.

Happy experimenting! 🚀
