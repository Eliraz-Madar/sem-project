# Running Part B with dataset.csv

## Overview
Part B of this project performs **document-level classification** to distinguish between factual and opinion articles. It uses a two-step approach:

1. **MCC (Multi-label Component Classifier)**: First trained on sentence-level data to identify argumentative components (claim, premise, other)
2. **Document Classifier**: Uses MCC predictions as features to classify entire articles as "fact" or "opinion"

## Your Dataset Format
Your `dataset.csv` contains:
- **ID**: Document identifier
- **category**: Label ("fact" or "opinion")
- **article_content**: The full text of the article

## Required Format for Part B
Part B expects JSONL (JSON Lines) format:
```json
{"doc_id": "0", "text": "Article text here...", "label": "fact", "publisher": "Unknown"}
{"doc_id": "1", "text": "Another article...", "label": "opinion", "publisher": "Unknown"}
```

## What Has Been Fixed

### 1. ✅ Conversion Script Created
**File**: `scripts/convert_dataset_csv_to_jsonl.py`

This script:
- Reads your `dataset.csv`
- Converts it to JSONL format
- Splits data into 80% training / 20% test
- Outputs to `src/data/news_train.jsonl` and `src/data/news_test.jsonl`

### 2. ✅ New Batch Script Created
**File**: `tools/run_part_b_with_dataset.cmd`

This script:
- Automatically converts your CSV to JSONL
- Runs Part B document classification
- Uses the converted data files

## How to Run Part B

### Option 1: Using the New Script (Recommended)
```cmd
tools\run_part_b_with_dataset.cmd
```

This will:
1. Convert `dataset.csv` to JSONL format
2. Run document classification on your data
3. Show results (macro-F1 scores for SVM and RNN models)

### Option 2: Manual Conversion Then Run
```cmd
# Step 1: Convert the dataset
python scripts\convert_dataset_csv_to_jsonl.py

# Step 2: Run Part B
tools\run_part_b_only.cmd
```

### Option 3: Custom Split Ratio
If you want a different train/test split, edit `convert_dataset_csv_to_jsonl.py` and change:
```python
convert_csv_to_jsonl(csv_file, train_jsonl, test_jsonl, split_ratio=0.8)  # Change 0.8 to desired ratio
```

## Prerequisites

Before running Part B, you need:

### 1. ✅ Trained MCC Model
You must have a trained MCC checkpoint at: `src/checkpoints/mcc_bert.pt`

**To train MCC (Part A)**, run:
```cmd
tools\run_all.cmd
```
This will:
- Set up the virtual environment
- Install dependencies
- Train the MCC model (sentence-level classifier)
- Train the document classifier

### 2. ✅ Virtual Environment
The virtual environment must be set up with all dependencies installed.

If not already done:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Expected Output

When you run Part B, you should see:
```
Converting dataset.csv...
Total documents loaded: 62951
Training documents: 50360
Test documents: 12591

Training label distribution:
  fact: 25180 (50.0%)
  opinion: 25180 (50.0%)

Running document classification...
SVM macro-F1 train=0.XXX test=0.XXX
Epoch 1 RNN macro-F1: 0.XXX
Epoch 2 RNN macro-F1: 0.XXX
Epoch 3 RNN macro-F1: 0.XXX
```

## Label Compatibility

Your dataset uses:
- ✅ **"fact"** - Maps to label 0 (news/factual)
- ✅ **"opinion"** - Maps to label 1 (opinion)

This is compatible with Part B, which expects exactly these two categories.

## Troubleshooting

### Error: "MCC checkpoint not found"
**Solution**: Run Part A first to train the MCC model:
```cmd
tools\run_all.cmd
```

### Error: "dataset.csv not found"
**Solution**: Make sure `dataset.csv` is in the root directory:
```
C:\Users\eliraz.eliezer.madar\Desktop\Sem-Project\sem-project\dataset.csv
```

### Error: "Virtual environment not found"
**Solution**: Create and activate the virtual environment:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Low Performance / Poor Results
This might happen because:
- MCC model needs more training data
- Your articles are very different from the training data used for MCC
- Consider training MCC on domain-specific data if available

## Files Modified/Created

1. ✅ **scripts/convert_dataset_csv_to_jsonl.py** - Converts CSV to JSONL
2. ✅ **tools/run_part_b_with_dataset.cmd** - Runs conversion + Part B
3. ✅ **DATASET_SETUP_GUIDE.md** - This documentation

## Summary

**Quick Start** (if environment is already set up):
```cmd
tools\run_part_b_with_dataset.cmd
```

**Full Setup** (first time):
```cmd
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Train MCC model (Part A)
tools\run_all.cmd

# 3. Run Part B with your dataset
tools\run_part_b_with_dataset.cmd
```
