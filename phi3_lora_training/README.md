# Phi-3 Medium LoRA Training Package

Complete end-to-end solution for fine-tuning Phi-3 Medium (14B parameters) using LoRA/QLoRA with custom datasets.

## Overview

This package provides:
- Dataset preparation pipeline (OpenHermes 2.5 + SlimOrca merging and cleaning)
- Environment setup with stable library versions
- LoRA training scripts with crash fixes
- Comprehensive documentation

## Quick Start

```bash
# 1. Setup environment (one-time)
./setup_training_env.sh

# 2. Prepare datasets (optional - if not already done)
./prepare_dataset.sh

# 3. Start training
./start_training.sh
```

## Directory Structure

```
phi3_lora_training/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies (pinned versions)
│
├── setup_training_env.sh              # Environment setup script
├── start_training.sh                  # Training launcher
├── prepare_dataset.sh                 # Dataset preparation launcher
│
├── scripts/
│   ├── prepare_dataset.py             # Dataset merging and cleaning
│   └── train_custom_dataset.py        # LoRA training script (with fixes)
│
└── docs/
    ├── README_TRAINING_SETUP.md       # Complete training guide
    ├── QUICK_START.md                 # 5-minute quick start
    ├── DATASET_PREPARATION.md         # Dataset pipeline details
    └── FILES_CREATED.md               # File reference guide
```

## What's Included

### Environment Setup
- **setup_training_env.sh**: Creates `labvenv` virtual environment with stable library versions
- **requirements.txt**: Pinned dependencies that fix compatibility issues
  - transformers==4.41.2 (NOT 4.57.x - prevents DynamicCache crashes)
  - accelerate==0.31.0 (NOT 1.12.x)
  - peft==0.11.1 (NOT 0.18.x)

### Dataset Preparation
- **prepare_dataset.sh**: Automates dataset download, merge, and cleaning
- **scripts/prepare_dataset.py**: Pipeline implementation
  - Downloads OpenHermes 2.5 and SlimOrca from HuggingFace
  - Normalizes conversation formats
  - Applies quality filters (character limits, turn counts)
  - Removes duplicates via SHA-256 hashing
  - Creates 90/10 train/validation split
  - Output: ~413k high-quality examples

### Training
- **start_training.sh**: Training launcher with proper configuration
- **scripts/train_custom_dataset.py**: LoRA training implementation
  - **Critical fix**: `model.config.use_cache = False` (prevents step 500 crash)
  - QLoRA with 4-bit quantization
  - LoRA rank 64, alpha 16
  - Sequence length 2048
  - Batch size 2 with gradient accumulation 4
  - Evaluation every 500 steps

### Documentation
- **README_TRAINING_SETUP.md**: Comprehensive guide with troubleshooting
- **QUICK_START.md**: Get started in under 5 minutes
- **DATASET_PREPARATION.md**: Detailed dataset pipeline documentation
- **FILES_CREATED.md**: Complete file reference

## Key Features

### 1. Fixed DynamicCache Crash
**Problem**: Training crashed at step 500 with `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`

**Solution**:
- Downgraded transformers to 4.41.2 (stable version)
- Added `model.config.use_cache = False` in training script
- Isolated environment (labvenv) prevents version conflicts

**Result**: Training completes all epochs without crashes

### 2. High-Quality Dataset
- **Sources**: OpenHermes 2.5 (300k) + SlimOrca (150k)
- **Quality filters**: Character limits, turn validation, role checks
- **Deduplication**: SHA-256 content hashing
- **Output**: 372,354 train + 41,372 validation examples
- **Avg quality**: ~1,492 chars/example, 2.5 turns/example

### 3. Efficient Training
- **QLoRA**: 4-bit quantization for memory efficiency
- **LoRA**: Only 0.6% parameters trainable
- **GPU**: Single GPU (20GB+ VRAM recommended)
- **Time**: ~65 hours for 3 epochs on consumer GPU

### 4. Complete Automation
- One-command environment setup
- One-command dataset preparation
- One-command training start
- Comprehensive error checking and validation

## Usage

### First Time Setup

```bash
cd /path/to/llm-lab-service/phi3_lora_training

# Create virtual environment and install dependencies
./setup_training_env.sh
```

### Prepare Dataset (if needed)

```bash
# Downloads and processes OpenHermes 2.5 + SlimOrca
./prepare_dataset.sh

# Output: ../scripts/datasets/merged_train.jsonl
#         ../scripts/datasets/merged_val.jsonl
```

### Start Training

```bash
# Launches training in background
./start_training.sh

# Monitor progress
tail -f /home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log

# Check GPU usage
nvidia-smi
```

## Configuration

Edit `start_training.sh` to customize training parameters:

```bash
--model_path /path/to/phi3-medium/          # Model location
--train_file ./path/to/train.jsonl         # Training data
--val_file ./path/to/val.jsonl             # Validation data
--output_dir /path/to/output/              # Where to save model
--max_seq_length 2048                       # Max sequence length
--batch_size 2                              # Per-device batch size
--gradient_accumulation_steps 4             # Effective batch = 8
--num_epochs 3                              # Training epochs
--learning_rate 2e-4                        # Learning rate
--lora_r 64                                 # LoRA rank
--lora_alpha 16                             # LoRA alpha
```

## Requirements

- **GPU**: NVIDIA GPU with 20GB+ VRAM
- **CUDA**: CUDA toolkit installed
- **Python**: 3.8+
- **Disk**: ~10GB free for datasets during preparation
- **Model**: Phi-3 Medium downloaded locally

## Troubleshooting

### DynamicCache Error at Step 500
**Fix**: Verify transformers version is 4.41.2
```bash
source labvenv/bin/activate
pip show transformers  # Should show 4.41.2
```

### Out of Memory
**Fix**: Reduce batch size in `start_training.sh`
```bash
--batch_size 1
```

### Import Errors
**Fix**: Ensure virtual environment is activated
```bash
source labvenv/bin/activate
```

## Advanced

### Custom Dataset Format

Your dataset should be JSONL with messages format:
```json
{
  "messages": [
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ]
}
```

### Using Different Models

Update `--model_path` in `start_training.sh` to point to any HuggingFace-compatible model.

### Evaluation After Training

The training includes evaluation every 500 steps. For additional evaluation:
```bash
# Use the evaluation scripts in parent directory
../scripts/evaluate_model.py
```

## Performance Metrics

**Expected training metrics**:
- Initial loss: ~1.0-1.1
- Final loss: ~0.8-0.9
- Training time: ~65 hours (3 epochs, consumer GPU)
- Memory usage: ~18GB VRAM with QLoRA

## Credits

- **OpenHermes 2.5**: teknium/OpenHermes-2.5 (MIT License)
- **SlimOrca**: Open-Orca/SlimOrca (MIT License)
- **Phi-3 Medium**: Microsoft (MIT License)

## License

This training package is provided under MIT license.
Component datasets retain their original licenses (all MIT).

## Support

For detailed documentation, see:
- `docs/README_TRAINING_SETUP.md` - Complete setup guide
- `docs/QUICK_START.md` - Quick start guide
- `docs/DATASET_PREPARATION.md` - Dataset details
- `docs/FILES_CREATED.md` - File reference

For issues with this training package, check the troubleshooting sections in the documentation files.
