# Dataset Preparation Guide

Complete guide for preparing high-quality training data by merging OpenHermes 2.5 and SlimOrca datasets.

## What This Does

This pipeline:
1. Downloads OpenHermes 2.5 and SlimOrca from HuggingFace
2. Normalizes conversation formats
3. Applies quality filters
4. Removes duplicates
5. Splits into train/validation sets
6. Outputs clean JSONL files ready for training

## Quick Start

```bash
cd /home/rnu/mnt/models/llm-lab-service/scripts
python prepare_dataset.py
```

**Time:** ~30-45 minutes
**Disk space needed:** ~5GB temporary, ~500MB final

---

## What You Get

After running, you'll have:

```
scripts/datasets/
├── merged_train.jsonl      # 372,354 training examples
├── merged_val.jsonl        # 41,372 validation examples
└── dataset_stats.json      # Quality metrics
```

**Total:** ~413,726 high-quality conversational examples

---

## Pipeline Steps Explained

### Step 1: Load OpenHermes 2.5

**Source:** `teknium/OpenHermes-2.5`
**Raw size:** ~1M examples
**Target size:** 300,000 examples

**What it contains:**
- Multi-turn conversations
- Diverse topics (coding, math, general knowledge)
- High-quality GPT-4 generated responses

### Step 2: Load SlimOrca

**Source:** `Open-Orca/SlimOrca`
**Raw size:** ~500k examples
**Target size:** 150,000 examples

**What it contains:**
- Instruction-following examples
- Reasoning tasks
- Complex multi-step problems

### Step 3: Normalize Format

Converts all conversations to standard format:
```json
{
  "messages": [
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ]
}
```

**Handles:**
- Different role names (human→user, gpt→assistant)
- System messages
- Missing or empty turns

### Step 4: Quality Filtering

**Filters applied:**

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Min characters | 50 | Remove too-short conversations |
| Max characters | 6,000 | Remove excessively long conversations |
| Min turns | 2 | Require at least 1 exchange |
| Role validation | Required | Must have both user and assistant |
| Empty check | None allowed | Remove empty messages |

**Typical retention rate:** ~85-90% of examples pass

### Step 5: Deduplication

**Method:** SHA-256 hash of conversation content

**What's removed:**
- Exact duplicates
- Similar examples from different sources

**Results:** Typically removes < 1% duplicates

### Step 6: Train/Validation Split

**Split ratio:** 90% train / 10% validation

| Set | Examples | Percentage |
|-----|----------|------------|
| Training | 372,354 | 90% |
| Validation | 41,372 | 10% |

**Shuffled:** Yes (seed=42 for reproducibility)

---

## Configuration Options

Edit `scripts/prepare_dataset.py` to customize:

```python
# Target sizes (adjust as needed)
TARGET_OH = 300_000      # OpenHermes samples
TARGET_SO = 150_000      # SlimOrca samples

# Quality filters
MAX_CHARS_PER_CONVO = 6000
MIN_CHARS_PER_CONVO = 50
MIN_TURNS = 2

# Split ratio
VAL_RATIO = 0.1  # 10% validation

# Random seed (for reproducibility)
RANDOM_SEED = 42
```

### Common Customizations

**Smaller dataset (faster training):**
```python
TARGET_OH = 100_000
TARGET_SO = 50_000
```

**Longer conversations only:**
```python
MIN_CHARS_PER_CONVO = 200
MIN_TURNS = 3
```

**No length limit:**
```python
MAX_CHARS_PER_CONVO = 99999
```

---

## Expected Output

### Console Output

```
============================================================
Loading OpenHermes 2.5...
============================================================
OpenHermes loaded: 1,002,137 examples
Normalizing conversations...
Applying quality filters...
After quality filter: 895,421 examples (89.4% kept)
Sampling to 300,000 examples...
Final OpenHermes: 300,000 examples

============================================================
Loading SlimOrca...
============================================================
SlimOrca loaded: 517,982 examples
Normalizing conversations...
Applying quality filters...
After quality filter: 462,154 examples (89.2% kept)
Sampling to 150,000 examples...
Final SlimOrca: 150,000 examples

============================================================
Merging datasets...
============================================================
Merged size before dedup: 450,000 examples

Deduplicating...
  Processed 0/450,000 examples...
  Processed 10,000/450,000 examples...
  ...
Removed 36,274 duplicates
Final merged size: 413,726 examples

============================================================
Creating train/validation split...
============================================================
Train: 372,354 examples (90%)
Val:   41,372 examples (10%)

============================================================
Saving datasets...
============================================================
Writing ./datasets/merged_train.jsonl...
Writing ./datasets/merged_val.jsonl...
Writing ./datasets/dataset_stats.json...

============================================================
SUMMARY
============================================================

Dataset Statistics:
  OpenHermes:    300,000 examples
  SlimOrca:      150,000 examples
  Merged:        413,726 examples
  Duplicates:     36,274 removed

Final Split:
  Training:      372,354 examples
  Validation:     41,372 examples

Quality Metrics:
  Avg chars/example:  1492
  Avg turns/example:  2.5

Output Files:
  ./datasets/merged_train.jsonl
  ./datasets/merged_val.jsonl
  ./datasets/dataset_stats.json

============================================================
Done!
============================================================
```

### Dataset Statistics File

`dataset_stats.json` contains:
```json
{
  "openhermes": {
    "name": "OpenHermes 2.5",
    "examples": 300000,
    "avg_chars_per_example": 1543.21,
    "avg_turns_per_example": 2.4,
    "role_distribution": {
      "user": 360000,
      "assistant": 360000,
      "system": 15000
    }
  },
  "slimorca": { ... },
  "merged": { ... },
  "train": { ... },
  "val": { ... }
}
```

---

## File Sizes

Approximate sizes:

| File | Size | Lines |
|------|------|-------|
| `merged_train.jsonl` | ~450 MB | 372,354 |
| `merged_val.jsonl` | ~50 MB | 41,372 |
| `dataset_stats.json` | ~2 KB | - |

**Total:** ~500 MB

---

## Quality Metrics Explained

### Average Characters per Example

**Typical range:** 1,000 - 2,000 characters

**What it means:**
- **< 500:** Very short exchanges (may lack detail)
- **500-1,000:** Concise conversations
- **1,000-2,000:** ✅ Good balance of detail and brevity
- **2,000-4,000:** Detailed explanations
- **> 4,000:** Very long conversations (may be too verbose)

### Average Turns per Example

**Typical range:** 2.0 - 3.0 turns

**What it means:**
- **2.0:** Single user question + assistant answer
- **2.5:** ✅ Mix of single and multi-turn conversations
- **3.0+:** More back-and-forth dialogue

### Role Distribution

Shows count of each role type:
- **user:** Questions/prompts
- **assistant:** Responses/answers
- **system:** System instructions (rare)

**Healthy ratio:** Approximately equal user/assistant counts

---

## Troubleshooting

### Issue: Out of memory during loading

**Solution:** Load datasets in streaming mode
```python
oh = load_dataset(OPENHERMES_NAME, split="train", streaming=True)
```

Then process in batches.

### Issue: Download failed

**Solution:** Check internet connection and HuggingFace access:
```bash
# Test HuggingFace connection
pip install huggingface_hub
python -c "from datasets import load_dataset; print('OK')"
```

### Issue: Taking too long

**Solutions:**
1. Reduce target sizes:
   ```python
   TARGET_OH = 100_000
   TARGET_SO = 50_000
   ```

2. Skip quality filters:
   ```python
   # Comment out filter step
   # oh_filtered = oh_normalized.filter(filter_quality)
   oh_filtered = oh_normalized
   ```

### Issue: Need more data

**Solution:** Increase target sizes:
```python
TARGET_OH = 500_000
TARGET_SO = 250_000
```

Or set to `None` to use all available data:
```python
TARGET_OH = None  # Use all OpenHermes
TARGET_SO = None  # Use all SlimOrca
```

---

## Using Custom Datasets

To add your own dataset:

1. **Load your dataset:**
```python
my_dataset = load_dataset("your/dataset", split="train")
```

2. **Normalize to messages format:**
```python
my_normalized = my_dataset.map(normalize_conversations)
```

3. **Merge with others:**
```python
merged = concatenate_datasets([oh_filtered, so_filtered, my_normalized])
```

---

## Verifying Dataset Quality

After preparation, verify your data:

```bash
# Check file sizes
ls -lh scripts/datasets/

# View first example
head -1 scripts/datasets/merged_train.jsonl | python -m json.tool

# Count examples
wc -l scripts/datasets/merged_train.jsonl
wc -l scripts/datasets/merged_val.jsonl

# View statistics
cat scripts/datasets/dataset_stats.json | python -m json.tool
```

### Sample Validation Example

```python
import json

# Read first example
with open("scripts/datasets/merged_train.jsonl") as f:
    example = json.loads(f.readline())

print("Example conversation:")
for msg in example["messages"]:
    print(f"{msg['role']}: {msg['content'][:100]}...")
```

Expected output:
```
Example conversation:
user: Explain quantum entanglement in simple terms...
assistant: Quantum entanglement is a phenomenon where two particles become...
```

---

## Next Steps

After dataset preparation:

1. **Verify files exist:**
   ```bash
   ls scripts/datasets/merged_*.jsonl
   ```

2. **Review statistics:**
   ```bash
   cat scripts/datasets/dataset_stats.json
   ```

3. **Start training:**
   ```bash
   ./start_training.sh
   ```

---

## Advanced: Custom Quality Filters

Add your own filters to `filter_quality()`:

```python
def filter_quality(example):
    msgs = example.get("messages") or []

    # Existing filters...
    if len(msgs) < MIN_TURNS:
        return False

    # NEW: Filter by language (example)
    text = " ".join(m["content"] for m in msgs)
    if detect_language(text) != "en":
        return False

    # NEW: Filter by topic (example)
    if "cryptocurrency" in text.lower():
        return False  # Skip crypto discussions

    # NEW: Require code in conversation (example)
    if not any("```" in m["content"] for m in msgs):
        return False  # Only keep examples with code

    return True
```

---

## Dataset Sources

### OpenHermes 2.5
- **License:** MIT
- **Link:** https://huggingface.co/datasets/teknium/OpenHermes-2.5
- **Quality:** GPT-4 generated, high quality
- **Size:** ~1M examples

### SlimOrca
- **License:** MIT
- **Link:** https://huggingface.co/datasets/Open-Orca/SlimOrca
- **Quality:** Filtered subset of OpenOrca
- **Size:** ~500k examples

---

## Reproducibility

The pipeline uses fixed random seed (42) for:
- Dataset shuffling
- Train/val split
- Sampling

**Same config = same output** every time

To use different random seed:
```python
RANDOM_SEED = 12345
```

---

## Performance Tips

1. **Use SSD for faster I/O** - Dataset operations are I/O intensive

2. **Close other programs** - Deduplication needs RAM (up to 4GB)

3. **Run overnight** - For full dataset, expect 30-60 minutes

4. **Monitor disk space** - Need 5-10GB free during processing

---

## FAQ

**Q: Can I use just one dataset (OpenHermes or SlimOrca)?**
A: Yes! Comment out the other dataset loading and merging sections.

**Q: How do I add more datasets?**
A: Load them, normalize format, filter, then add to `concatenate_datasets()`.

**Q: Can I skip deduplication?**
A: Yes, but not recommended. Comment out the deduplication section.

**Q: What if I want different train/val split?**
A: Change `VAL_RATIO = 0.1` to desired percentage (0.2 = 20% val).

**Q: Can I use this for non-English data?**
A: Yes! Remove or modify the language filtering if present.

---

## Support

For issues with dataset preparation:

1. Check disk space: `df -h`
2. Check memory: `free -h`
3. Review error messages in console
4. Verify internet connection for downloads
5. Check HuggingFace dataset availability

---

## License

This preparation script is provided under MIT license.
Datasets retain their original licenses (both MIT).
