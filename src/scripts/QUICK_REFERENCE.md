# Pipeline Tracer - Quick Reference

## Installation & Setup

The trace pipeline script is ready to use. No additional dependencies are required beyond the existing project requirements.

## Three Example Commands

### 1. Dry-Run (File Check Only)
Verify all required files exist without running the pipeline:

```powershell
cd src
python -m scripts.trace_pipeline --dry-run
```

**What it does:** 
- Checks if training data, test data, and MCC checkpoint exist
- Validates checkpoint is loadable
- Prints file sizes and basic info
- **Does NOT run** any model training or inference

**Use when:** You want to verify setup before a long run.

---

### 2. Smoke Test (Quick Validation)
Run the full pipeline on a tiny subset (20 documents):

```powershell
cd src
python -m scripts.trace_pipeline --smoke
```

**What it does:**
- Uses only first 20 training docs and 20 test docs
- Runs all three stages (A, B, C)
- Completes in ~2-5 minutes (depending on hardware)
- Saves trace report to `outputs/trace/`

**Use when:** You want quick end-to-end validation during development.

---

### 3. Full Pipeline
Run the complete pipeline with all data:

```powershell
cd src
python -m scripts.trace_pipeline --mcc-checkpoint checkpoints/mcc_webis_tiny.pt
```

**What it does:**
- Processes all training and test documents
- Runs sentence-level MCC classification
- Extracts document features (ratios + sequences)
- Trains SVM (baseline) and RNN (argumentation) classifiers
- Reports Accuracy, Macro-F1, Confusion Matrix
- Saves detailed timing report

**Use when:** Final validation or preparing results for submission.

---

## Output Interpretation

### Console Output
The script prints clear stage banners with timing:

```
================================================================================
  STAGE A: SENTENCE-LEVEL ARGUMENT MINING
  Loading BERT-based MCC classifier (claim/premise/other)
================================================================================
⏱  Started at: 2026-01-05 14:30:01
⏱  Elapsed since start: 0.15s

🤖 Loading MCC model: bert-base-uncased
✓ Model loaded successfully
   Device: cuda
   Num labels: 3

⏱  Stage completed in: 3.42s
⏱  Total elapsed: 3.57s
```

### Sample Document Analysis
For the first 3 documents, you'll see:
- Sentence-by-sentence predictions with labels
- Aggregated ratio features
- Example of how claims/premises are distributed

### Final Metrics
```
📊 Overall Results:
   SVM (Baseline):  Test F1 = 0.7891, Acc = 0.8125
   RNN (Argument):  Test F1 = 0.8012

   Confusion Matrix:
             news  opinion
   news       45        8
   opinion    12       51
```

---

## Files Generated

After running, check `outputs/trace/`:

```
outputs/trace/
├── trace_report.json    # Detailed timing and metadata for each stage
└── results.json         # Model performance metrics (F1, accuracy, confusion matrix)
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Training data NOT FOUND` | Ensure you're in `src/` directory. Check if `data/news_train.jsonl` exists. |
| `MCC checkpoint corrupted` | Re-train: `python -m scripts.run_mcc_training` |
| `Out of memory` | Use `--smoke` flag or reduce `--batch-size 2` |
| `ImportError: No module named transformers` | Install: `pip install -r ../requirements.txt` |

---

## Advanced Usage

### Custom Paths
```powershell
python -m scripts.trace_pipeline `
    --news-train custom_data/my_train.jsonl `
    --news-test custom_data/my_test.jsonl `
    --mcc-checkpoint models/my_checkpoint.pt `
    --out-dir results/experiment_001
```

### Smaller Smoke Test
```powershell
python -m scripts.trace_pipeline --smoke --smoke-size 5
```

### Adjust RNN Training
```powershell
python -m scripts.trace_pipeline --epochs 5 --batch-size 8 --lr 0.001
```

---

## Expected Runtime

| Mode | Documents | Estimated Time |
|------|-----------|----------------|
| Dry-run | 0 | <1 second |
| Smoke (20 docs) | 40 total | 2-5 minutes |
| Full (default) | ~200-500 | 10-30 minutes |

*Times assume CPU inference. GPU is 3-5x faster for Stage A/B.*

---

## Assumptions

The script assumes:
1. Working directory is `src/` when running
2. Documents have fields: `text`, `label` (required); `doc_id`, `publisher` (optional)
3. Labels are "news" or "opinion" (case-insensitive)
4. MCC checkpoint is compatible with `--model-name` (default: `bert-base-uncased`)

---

## Integration

This script **reuses existing code** from:
- `mcc.models` - BERT MCC classifier
- `doc_clf.features` - Feature extraction pipeline
- `doc_clf.models` - SVM and RNN classifiers
- `doc_clf.train` - Training utilities

No code duplication. It's a tracing wrapper around the existing implementation.
