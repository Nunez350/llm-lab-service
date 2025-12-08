# Dataset Filtering and Quality Control

This document describes the dataset filtering and quality control tools used to prepare the training dataset.

## Overview

The training dataset has been filtered to:
1. **Remove non-English sequences** - Only English content is used for training
2. **Identify long sequences** - Sequences exceeding `max_seq_length` are identified for review

## English-Only Filtering

### Problem
The original merged dataset contained ~1.5% non-English sequences (6,201 out of 413,726 total sequences), including:
- French, Turkish, Romanian, Spanish, Finnish, German, Russian, Czech, Chinese, and others

### Solution
We use the `detect_non_english.py` script to identify and filter out non-English sequences.

### Usage

**Analyze dataset for non-English sequences:**
```bash
python scripts/detect_non_english.py \
    --train_file scripts/datasets/merged_train.jsonl \
    --val_file scripts/datasets/merged_val.jsonl
```

**Filter out non-English sequences and create English-only datasets:**
```bash
python scripts/detect_non_english.py \
    --train_file scripts/datasets/merged_train.jsonl \
    --val_file scripts/datasets/merged_val.jsonl \
    --filter \
    --output_dir scripts/datasets
```

This creates:
- `scripts/datasets/merged_train_english_only.jsonl` (366,765 sequences)
- `scripts/datasets/merged_val_english_only.jsonl` (40,760 sequences)

### Results

**Before filtering:**
- Training: 372,354 sequences (5,605 non-English = 1.51%)
- Validation: 41,372 sequences (617 non-English = 1.49%)
- **Total: 413,726 sequences (6,222 non-English = 1.51%)**

**After filtering:**
- Training: 366,765 English-only sequences
- Validation: 40,760 English-only sequences
- **Total: 407,525 English-only sequences**

**Removed:** 6,201 non-English sequences (1.51% of dataset)

## Sequence Length Analysis

### Tools

1. **check_sequence_lengths.py** - Analyzes entire dataset for sequence length statistics
2. **find_long_sequences.py** - Finds and saves sequences exceeding max_seq_length for review

See individual script help for usage: `python scripts/check_sequence_lengths.py --help`

### Current Statistics (English-Only Dataset)

- 99.7% of sequences are within 1536 tokens
- Only 0.30% exceed the threshold (will be truncated during training)
- This is acceptable given the small percentage

## Training Configuration

The training script uses English-only datasets by default:
- Training: `scripts/datasets/merged_train_english_only.jsonl`
- Validation: `scripts/datasets/merged_val_english_only.jsonl`

## Dependencies

English filtering requires: `pip install langdetect`
