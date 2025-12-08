# Reproduction Guide - Full bf16 Training Setup

Complete step-by-step guide to reproduce the full bf16 training configuration.

## Quick Summary

**Current Working Configuration:**
- Model: Phi-3 Medium (14B parameters)
- Precision: Full bf16 (no quantization)
- Batch size: 1 (memory-optimized)
- Gradient accumulation: 64
- Effective batch size: 64
- Max sequence length: 1536
- Training time: ~93 hours (3.9 days)
- GPU: RTX 5090 (32GB VRAM)

## Step-by-Step Reproduction

### 1. Clone and Navigate

```bash
cd /home/rnu/mnt/models/llm-lab-service
git checkout feature/lora_training_13b
cd phi3_lora_training
```

### 2. Setup Environment

```bash
./setup_training_env.sh
```

### 3. Verify Environment

```bash
source ../labvenv/bin/activate
pip show transformers  # Must show 4.41.2
pip show peft          # Must show 0.11.1
```

### 4. Prepare Dataset

```bash
./prepare_dataset.sh
```

### 5. Filter English-Only Sequences (Recommended)

The dataset contains ~1.5% non-English sequences. Filter them out:

```bash
cd ..
python scripts/detect_non_english.py \
    --train_file scripts/datasets/merged_train.jsonl \
    --val_file scripts/datasets/merged_val.jsonl \
    --filter \
    --output_dir scripts/datasets
```

This creates English-only datasets that will be used for training.

### 6. Start Training

```bash
cd phi3_lora_training
./start_training.sh
```

Or manually:
```bash
cd /home/rnu/mnt/models/llm-lab-service
source labvenv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python scripts/train_custom_dataset.py \
  --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
  --train_file scripts/datasets/merged_train.jsonl \
  --val_file scripts/datasets/merged_val.jsonl \
  --output_dir /home/rnu/mnt/models/unsloth/phi3-finetuned-stable-bf16 \
  --max_seq_length 1536 \
  --batch_size 1 \
  --gradient_accumulation_steps 64 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 64 \
  --lora_alpha 16 \
  --gpu_id 0 \
  --eval_steps 5000 \
  --dataloader_num_workers 2 \
  --dataloader_pin_memory True \
  > /home/rnu/mnt/models/unsloth/phi3-finetuned-stable-bf16/training.log 2>&1 &
```

## Key Code Changes

1. **Conditional bitsandbytes import** (Line 23, 186-188)
2. **Flash Attention disabled** (Lines 175-181)
3. **DynamicCache compatibility** (Lines 29-40, 210-219)

See `TRAINING_GUIDE_BF16.md` for detailed documentation.
