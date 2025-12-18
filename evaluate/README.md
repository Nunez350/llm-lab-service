# LLM Evaluation Framework

Two-GPU evaluation framework for comparing fine-tuned LoRA adapters against base models.

## Features

- **Perplexity comparison**: Quantitative metric showing model improvement
- **A/B comparison**: Side-by-side response generation from both models
- **Sample generation**: Generate responses with ground truth comparison
- **Two-GPU mode**: Runs fine-tuned on GPU 0, base on GPU 1 for efficiency

## Quick Start

```bash
cd /home/rnu/mnt/models/llm-lab-service/evaluate
./run_evaluation.sh
```

## Configuration

Override defaults with environment variables:

```bash
# Custom checkpoint
MODEL_PATH=/path/to/checkpoint ./run_evaluation.sh

# All options
MODEL_PATH=/path/to/adapter \
BASE_MODEL_PATH=/path/to/base \
VAL_FILE=/path/to/val.jsonl \
OUTPUT_DIR=/path/to/output \
NUM_SAMPLES=200 \
./run_evaluation.sh
```

## Default Configuration

| Parameter | Default |
|-----------|---------|
| MODEL_PATH | `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-22000` |
| BASE_MODEL_PATH | `/home/rnu/mnt/models/phi_models/phi3-medium/` |
| VAL_FILE | `../scripts/datasets/merged_val_english_only_short.jsonl` |
| OUTPUT_DIR | `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/evaluation` |
| NUM_SAMPLES | 100 |

## Output Files

Each run generates timestamped files in OUTPUT_DIR:

| File | Description |
|------|-------------|
| `perplexity_TIMESTAMP.json` | Loss and perplexity for both models |
| `samples_TIMESTAMP.json` | Generated responses with ground truth |
| `ab_comparison_TIMESTAMP.json` | Side-by-side base vs fine-tuned responses |
| `summary_TIMESTAMP.md` | Human-readable summary report |

## GPU Requirements

- **GPU 0**: ~28GB VRAM (fine-tuned model in bfloat16)
- **GPU 1**: ~28GB VRAM (base model in bfloat16)
- Total: 2x 32GB GPUs recommended

## Example Results

From checkpoint-22000 evaluation:

| Model | Loss | Perplexity |
|-------|------|------------|
| Fine-tuned | 0.7387 | 2.0933 |
| Base | 1.0422 | 2.8356 |

**Improvement: 26.2% lower perplexity**

## Files

```
evaluate/
├── evaluate_model.py   # Main evaluation script
├── run_evaluation.sh   # Launcher script
└── README.md           # This file
```

## Usage Examples

### Evaluate a specific checkpoint

```bash
MODEL_PATH=/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-20000 \
./run_evaluation.sh
```

### Increase sample size for more accurate perplexity

```bash
NUM_SAMPLES=500 ./run_evaluation.sh
```

### Custom output directory

```bash
OUTPUT_DIR=/tmp/eval_results ./run_evaluation.sh
```
