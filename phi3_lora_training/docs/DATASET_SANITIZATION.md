# Comprehensive Dataset Sanitization Guide

This document describes all available dataset sanitization and quality control measures.

## Overview

We've identified several issues in datasets:
1. **Different languages** - ~1.5% non-English sequences
2. **Different sizes** - Sequences ranging from very short to very long
3. **Format issues** - Malformed messages, missing roles, etc.
4. **Quality issues** - Repetition, encoding errors, empty content

## Available Tools

### 1. Language Filtering (`detect_non_english.py`)

**Purpose:** Remove non-English sequences

**Usage:**
```bash
python scripts/detect_non_english.py \
    --train_file scripts/datasets/merged_train.jsonl \
    --val_file scripts/datasets/merged_val.jsonl \
    --filter \
    --output_dir scripts/datasets
```

**What it does:**
- Detects language of each sequence using `langdetect`
- Filters out non-English sequences
- Creates English-only datasets

**Results:** Removed 6,201 non-English sequences (1.51% of dataset)

### 2. Sequence Length Analysis (`check_sequence_lengths.py`)

**Purpose:** Analyze sequence length distribution

**Usage:**
```bash
python scripts/check_sequence_lengths.py \
    --model_path microsoft/Phi-3-medium-4k-instruct \
    --train_file scripts/datasets/merged_train_english_only.jsonl \
    --val_file scripts/datasets/merged_val_english_only.jsonl \
    --max_seq_length 1536
```

**What it does:**
- Tokenizes sequences without truncation
- Provides statistics (min, max, mean, median, percentiles)
- Shows how many sequences exceed max_seq_length
- Displays length distribution

### 3. Find Long Sequences (`find_long_sequences.py`)

**Purpose:** Identify sequences exceeding max_seq_length for review

**Usage:**
```bash
python scripts/find_long_sequences.py \
    --model_path microsoft/Phi-3-medium-4k-instruct \
    --train_file scripts/datasets/merged_train_english_only.jsonl \
    --val_file scripts/datasets/merged_val_english_only.jsonl \
    --max_seq_length 1536 \
    --output_file scripts/datasets/long_sequences.jsonl
```

**What it does:**
- Finds all sequences exceeding the threshold
- Saves them to a file with full details for manual review
- Includes token counts, excess amounts, and full conversations

### 4. Comprehensive Sanitization (`sanitize_dataset.py`) ⭐ NEW

**Purpose:** Apply multiple quality filters in one pass

**Usage:**
```bash
python scripts/sanitize_dataset.py \
    --train_file scripts/datasets/merged_train_english_only.jsonl \
    --val_file scripts/datasets/merged_val_english_only.jsonl \
    --output_dir scripts/datasets \
    --model_path microsoft/Phi-3-medium-4k-instruct \
    --filter_language \
    --min_tokens 10 \
    --max_tokens 1536 \
    --min_turns 2 \
    --max_repetition_ratio 0.5
```

## Sanitization Checks

The comprehensive sanitizer performs these checks:

### 1. Format Validation
- ✅ Messages must be proper dictionaries
- ✅ Each message must have "role" and "content" fields
- ✅ Content must be a string
- ✅ Must have at least one user message
- ✅ Must have at least one assistant message

### 2. Content Quality
- ✅ Content must not be empty
- ✅ Minimum length check (default: 20 characters)
- ✅ Maximum whitespace check (not >50% whitespace)
- ✅ Minimum conversation turns (default: 2)
- ✅ User messages must have responses

### 3. Language Filtering
- ✅ Detects language using `langdetect`
- ✅ Filters out non-English sequences (if `--filter_language` enabled)

### 4. Token Length Validation
- ✅ Minimum token count (default: 10 tokens)
- ✅ Maximum token count (default: 1536 tokens)
- ✅ Uses actual tokenizer for accurate counting

### 5. Encoding Validation
- ✅ Checks for replacement characters (encoding errors)
- ✅ Checks for excessive control characters (>1% of text)

### 6. Repetition Detection
- ✅ Calculates sentence-level repetition ratio
- ✅ Detects repeated n-grams (3-word phrases)
- ✅ Filters sequences with >50% repetition (configurable)

## Recommended Sanitization Pipeline

### Step 1: Basic Language Filtering
```bash
python scripts/detect_non_english.py \
    --train_file scripts/datasets/merged_train.jsonl \
    --val_file scripts/datasets/merged_val.jsonl \
    --filter \
    --output_dir scripts/datasets
```

### Step 2: Comprehensive Sanitization
```bash
python scripts/sanitize_dataset.py \
    --train_file scripts/datasets/merged_train_english_only.jsonl \
    --val_file scripts/datasets/merged_val_english_only.jsonl \
    --output_dir scripts/datasets \
    --model_path microsoft/Phi-3-medium-4k-instruct \
    --filter_language \
    --min_tokens 10 \
    --max_tokens 1536 \
    --min_turns 2 \
    --max_repetition_ratio 0.5
```

This creates:
- `merged_train_english_only_sanitized.jsonl`
- `merged_val_english_only_sanitized.jsonl`

### Step 3: Review Long Sequences (Optional)
```bash
python scripts/find_long_sequences.py \
    --model_path microsoft/Phi-3-medium-4k-instruct \
    --train_file scripts/datasets/merged_train_english_only_sanitized.jsonl \
    --val_file scripts/datasets/merged_val_english_only_sanitized.jsonl \
    --max_seq_length 1536 \
    --output_file scripts/datasets/long_sequences.jsonl
```

Review `long_sequences.jsonl` to decide if any should be removed or split.

## Filter Statistics

After running comprehensive sanitization, you'll see:
- Total sequences processed
- Number passed vs filtered
- Breakdown of filter reasons:
  - `non_english_*` - Non-English sequences
  - `too_short_tokens_*` - Below minimum token count
  - `too_long_tokens_*` - Above maximum token count
  - `excessive_repetition_*` - High repetition ratio
  - `encoding_errors` - Encoding issues detected
  - `excessive_control_chars` - Too many control characters
  - `empty_content` - Empty or whitespace-only content
  - `too_short` - Below minimum character count
  - `insufficient_turns` - Not enough conversation turns
  - `no_response_to_user` - User message without response
  - `json_decode_error` - Invalid JSON format

## Customization

### Adjust Token Limits
```bash
--min_tokens 20      # Increase minimum (default: 10)
--max_tokens 2048    # Increase maximum (default: 1536)
```

### Adjust Repetition Threshold
```bash
--max_repetition_ratio 0.3  # Stricter (default: 0.5)
--max_repetition_ratio 0.7  # More lenient
```

### Adjust Turn Requirements
```bash
--min_turns 3  # Require more conversation turns (default: 2)
```

## Dependencies

```bash
pip install langdetect transformers
```

## Performance

- Processing speed: ~10,000-15,000 sequences/second
- Memory usage: Minimal (streaming processing)
- For 400K sequences: ~30-60 seconds

## Best Practices

1. **Run language filtering first** - Removes obvious non-English content
2. **Then run comprehensive sanitization** - Catches format, quality, and encoding issues
3. **Review long sequences manually** - Decide if they should be kept, removed, or split
4. **Check filter statistics** - Understand what's being filtered and why
5. **Iterate on thresholds** - Adjust based on your dataset characteristics

## Example Output

```
============================================================
Dataset Sanitization Tool
============================================================
Filters enabled:
  - Language filter: Yes (English only)
  - Token range: 10 - 1536
  - Min turns: 2
  - Max repetition: 0.5
============================================================

📊 Results for TRAIN:
  Total sequences: 366,765
  Passed: 360,123 (98.19%)
  Filtered: 6,642 (1.81%)

  Filter reasons:
    too_long_tokens_*: 1,116
    excessive_repetition_*: 892
    non_english_*: 234
    encoding_errors: 156
    ...
```

This gives you full visibility into dataset quality and allows you to make informed decisions about what to keep or filter.
