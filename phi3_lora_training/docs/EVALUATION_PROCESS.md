# Evaluation Process Documentation

**Date:** December 10, 2025  
**Model:** Phi-3 Medium Fine-tuned (checkpoint-22000)

## Overview

This document describes the evaluation process for the fine-tuned Phi-3 Medium model, including the two-GPU setup and dataset information.

---

## Evaluation Setup

### Two-GPU Configuration

The evaluation uses a two-GPU setup to efficiently compare the base model and fine-tuned model simultaneously:

- **GPU 0:** Fine-tuned model (with LoRA adapter)
- **GPU 1:** Base model (for comparison)

This allows parallel evaluation without memory conflicts.

### Command Used

```bash
cd /home/rnu/mnt/models/llm-lab-service/evaluate
./run_evaluation.sh
```

Or with a specific checkpoint:

```bash
MODEL_PATH=/path/to/checkpoint ./run_evaluation.sh
```

---

## How the Evaluation Works

### 1. Model Loading (Two GPUs)

#### Fine-tuned Model on GPU 0

```python
# Load base model on GPU 0
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/rnu/mnt/models/phi_models/phi3-medium/",
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # Force to GPU 0
    trust_remote_code=True
)

# Load LoRA adapter
finetuned_model = PeftModel.from_pretrained(
    base_model,
    "/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-22000"
)
```

#### Base Model on GPU 1

```python
# Load base model on GPU 1
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/rnu/mnt/models/phi_models/phi3-medium/",
    torch_dtype=torch.bfloat16,
    device_map={"": 1},  # Force to GPU 1
    trust_remote_code=True
)
```

### 2. Perplexity Calculation

- Loads validation samples (default: 100 samples)
- For each sample:
  1. Tokenize the text
  2. Forward pass through model
  3. Calculate loss
- Perplexity = exp(average_loss)

**Lower perplexity = better model performance**

### 3. Sample Generation

- Generates responses from both models
- Uses same prompts for fair comparison
- Saves samples for qualitative review

### 4. A/B Comparison

- Same prompt sent to both models simultaneously
- Each model generates on its own GPU
- Responses saved side-by-side for comparison
- Enables direct quality assessment

---

## Key Files

### Scripts

- **Evaluation Script:** `/home/rnu/mnt/models/llm-lab-service/evaluate/evaluate_model.py`
- **Launcher:** `/home/rnu/mnt/models/llm-lab-service/evaluate/run_evaluation.sh`

### Results

- **Output Directory:** `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/evaluation/`

**Output Files:**
- `perplexity_*.json` - Perplexity metrics for both models
- `samples_*.json` - Generated sample responses
- `ab_comparison_*.json` - Side-by-side comparison data
- `summary_*.md` - Human-readable evaluation summary

---

## Running Evaluation

### Default (Latest Checkpoint)

```bash
cd /home/rnu/mnt/models/llm-lab-service/evaluate
./run_evaluation.sh
```

### Specific Checkpoint

```bash
MODEL_PATH=/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-20000 ./run_evaluation.sh
```

### Custom Parameters

```bash
cd /home/rnu/mnt/models/llm-lab-service/evaluate

python evaluate_model.py \
  --model_path /path/to/checkpoint \
  --base_model_path /home/rnu/mnt/models/phi_models/phi3-medium \
  --val_file datasets/merged_val_english_only_short.jsonl \
  --output_dir ./evaluation_results \
  --num_samples 100 \
  --gpu_id 0
```

---

## Training Datasets

### Model Training Data

The fine-tuned model (checkpoint-22000) was trained on:

#### Dataset Files

- **Training:** `scripts/datasets/merged_train_english_only_short.jsonl`
- **Validation:** `scripts/datasets/merged_val_english_only_short.jsonl`

#### Source Datasets (Merged)

1. **OpenHermes 2.5**
   - High-quality instruction-following conversations
   - Diverse task types and domains

2. **SlimOrca**
   - Cleaned version of OpenOrca dataset
   - Instruction-response pairs

#### Filtering Applied

1. **English-only filtering**
   - Non-English conversations removed
   - Language detection using `langdetect` library

2. **Length filtering**
   - Sequences filtered to fit `max_seq_length=1024`
   - Short sequences version created for faster training

3. **Quality filters**
   - Character limits enforced
   - Turn validation (proper role sequences)
   - Role checks (system/user/assistant)

4. **Deduplication**
   - SHA-256 content hashing
   - Duplicate conversations removed

#### Dataset Statistics

- **Training samples:** ~372K (full), varies for short version
- **Validation samples:** ~41K (full), varies for short version
- **Average conversation length:** ~1,492 characters
- **Average turns per conversation:** ~2.5

#### Data Format

The data is in **chat/messages format** with structured turns:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

This format is why the fine-tuned model improved at instruction-following tasks.

---

## Evaluation Metrics

### Perplexity

- **Definition:** exp(average_loss)
- **Interpretation:** Lower is better
- **Comparison:** Fine-tuned vs base model perplexity

### Sample Quality

- **Helpfulness:** Does it address the user's question?
- **Accuracy:** Is the information correct?
- **Coherence:** Is it well-structured?
- **Relevance:** Does it stay on topic?

### A/B Comparison

- Side-by-side response comparison
- Direct quality assessment
- Win rate calculation (fine-tuned vs base)

---

## Notes

- **GPU Memory:** Two-GPU setup prevents OOM errors
- **Checkpoint Selection:** checkpoint-22000 is the latest, checkpoint-20000 has best eval_loss
- **Validation File:** Using short version for faster evaluation
- **Model Format:** Fine-tuned model uses LoRA adapters, merged during evaluation

---

## Future Enhancements

1. **LLM-as-Judge:** Add local LLM support for automated quality scoring
2. **Benchmark Testing:** Add standard benchmarks (MMLU, HellaSwag, etc.)
3. **Human Evaluation:** Collect human ratings for quality assessment
4. **Automated Reporting:** Generate comparison reports automatically

---

**Last Updated:** December 10, 2025

