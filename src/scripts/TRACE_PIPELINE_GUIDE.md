# Pipeline Tracer Guide

The `trace_pipeline.py` script provides comprehensive end-to-end validation of the seminar project's implementation with detailed logging and timing.

## What It Does

The tracer validates all three stages of the pipeline:

1. **STAGE A: Sentence-Level Argument Mining**
   - Loads the pre-trained BERT MCC classifier
   - Verifies it can classify sentences into claim/premise/other
   - Performs sanity checks with test sentences

2. **STAGE B: Document-Level Feature Aggregation**
   - Processes each document by splitting into sentences
   - Classifies each sentence using the MCC model
   - Aggregates predictions into ratio features (e.g., `ratio_claim`, `ratio_premise`)
   - Extracts tag sequences for sequential models
   - Displays sample predictions for first 3 documents

3. **STAGE C: Document Classifier Training & Evaluation**
   - **C.1 Baseline (Lexical)**: Trains SVM on ratio features
   - **C.2 Argumentation-Based**: Trains GRU-RNN on tag sequences
   - Reports Accuracy, Macro-F1, and Confusion Matrix

## Usage

### Basic Commands

```powershell
# Full pipeline with default paths (from src/ directory)
cd src
python -m scripts.trace_pipeline

# Quick smoke test with only 20 documents
python -m scripts.trace_pipeline --smoke

# Dry-run: check files exist without running
python -m scripts.trace_pipeline --dry-run

# Custom paths
python -m scripts.trace_pipeline `
    --news-train data/news_train.jsonl `
    --news-test data/news_test.jsonl `
    --mcc-checkpoint checkpoints/mcc_webis_tiny.pt `
    --out-dir outputs/trace_run_001
```

### Command-Line Arguments

**Input Files:**
- `--news-train PATH`: Training documents (default: `data/news_train.jsonl`)
- `--news-test PATH`: Test documents (default: `data/news_test.jsonl`)
- `--mcc-checkpoint PATH`: Pre-trained MCC model (default: `checkpoints/mcc_webis_tiny.pt`)
- `--model-name NAME`: BERT model name (default: `bert-base-uncased`)

**Output:**
- `--out-dir PATH`: Directory for trace reports (default: `outputs/trace`)

**Training Parameters:**
- `--epochs N`: RNN training epochs (default: 3)
- `--batch-size N`: Batch size for RNN (default: 4)
- `--lr FLOAT`: Learning rate for RNN (default: 1e-3)

**Modes:**
- `--smoke`: Smoke test mode (uses first 20 docs only)
- `--smoke-size N`: Number of docs in smoke test (default: 20)
- `--dry-run`: Only check files, don't run pipeline

## Output Files

The script generates two JSON files in the output directory:

### 1. `trace_report.json`
Contains detailed timing and metadata for each stage:
```json
{
  "pipeline_start": "2026-01-05 14:30:00",
  "pipeline_end": "2026-01-05 14:35:42",
  "total_elapsed_seconds": 342.5,
  "stages": [
    {
      "stage_name": "FILE VERIFICATION",
      "elapsed_seconds": 0.15,
      "start_timestamp": "2026-01-05 14:30:00",
      "end_timestamp": "2026-01-05 14:30:00",
      "metadata": {...}
    },
    ...
  ]
}
```

### 2. `results.json`
Contains final model performance metrics:
```json
{
  "svm": {
    "train_f1": 0.8234,
    "test_f1": 0.7891,
    "test_accuracy": 0.8125,
    "confusion_matrix": [[45, 8], [12, 51]]
  },
  "rnn": {
    "final_test_f1": 0.8012,
    "epoch_f1s": [0.7234, 0.7823, 0.8012]
  }
}
```

## Example Output

```
================================================================================
  SEMINAR PROJECT: PIPELINE TRACER
  End-to-end validation of argument-based news classification
================================================================================
📅 Started at: 2026-01-05 14:30:00
💻 Device: CUDA
🚀 Mode: FULL PIPELINE

================================================================================
  FILE VERIFICATION
  Checking input files and dependencies
================================================================================
⏱  Started at: 2026-01-05 14:30:00
⏱  Elapsed since start: 0.00s

✓ Training data: data/news_train.jsonl (0.25 MB)
✓ Test data: data/news_test.jsonl (0.08 MB)
✓ MCC checkpoint: checkpoints/mcc_webis_tiny.pt (438.21 MB)
✓ MCC checkpoint loadable (218 state dict keys)

⏱  Stage completed in: 0.15s
⏱  Total elapsed: 0.15s

================================================================================
  STAGE A: SENTENCE-LEVEL ARGUMENT MINING
  Loading BERT-based MCC classifier (claim/premise/other)
================================================================================
⏱  Started at: 2026-01-05 14:30:01
⏱  Elapsed since start: 0.15s

🤖 Loading MCC model: bert-base-uncased
📦 Loading checkpoint: checkpoints/mcc_webis_tiny.pt
✓ Model loaded successfully
   Device: cuda
   Num labels: 3
   Labels: ('claim', 'premise', 'other')

🧪 Sanity check with 3 test sentences:
   'The sky is blue.' → claim
   'This proves my point.' → premise
   'Hello world.' → other

⏱  Stage completed in: 3.42s
⏱  Total elapsed: 3.57s

...
```

## Troubleshooting

### Missing Files
If you see `❌ Training data NOT FOUND`:
- Check the path is correct
- From project root: ensure you're in the `src/` directory when running
- Run data preparation: `python -m scripts.convert_webis_to_jsonl`

### Checkpoint Incompatible
If you see `❌ MCC checkpoint corrupted or incompatible`:
- Re-train the MCC model: `python -m scripts.run_mcc_training`
- Ensure the checkpoint matches the `--model-name` argument

### Out of Memory
- Use `--smoke` flag to test with fewer documents
- Reduce `--batch-size` (default: 4)
- Switch to CPU if GPU memory is insufficient (automatic fallback)

## Assumptions

The script assumes:
1. You're running from the `src/` directory
2. Data files are in JSONL format with fields: `text`, `label`, `doc_id`, `publisher` (optional)
3. MCC checkpoint is compatible with the specified `--model-name`
4. Labels in documents are either "news" or "opinion" (case-insensitive)

## Integration with Existing Code

The tracer **reuses** existing modules:
- `mcc.models.BertMCCClassifier`: MCC model wrapper
- `doc_clf.features.extract_features_for_corpus`: Feature extraction
- `doc_clf.models.train_svm`, `ArgumentRNNClassifier`: Classifiers
- `doc_clf.train`: Training and evaluation utilities

No duplicate logic is implemented.
