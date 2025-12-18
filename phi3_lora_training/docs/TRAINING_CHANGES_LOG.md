# Training Configuration Changes Log

**Date:** December 9, 2025  
**Last Updated:** December 9, 2025  
**Session:** Phi-3 Medium Fine-tuning Optimization

## Summary

This document tracks all changes made to optimize Phi-3 Medium fine-tuning for multi-GPU training with QLoRA (4-bit quantization).

---

## Key Changes Overview

### 1. Sequence Length Reduction
- **Initial:** 2048 tokens
- **Intermediate:** 1536 tokens
- **Final:** 1026 tokens (in `train_working.py`), 1024 tokens (in `train_custom_dataset.py`)
- **Reason:** Reduce memory usage and speed up training
- **Impact:** ~33% fewer tokens per sample, faster training

### 2. Evaluation Frequency
- **Initial:** Every 5000 steps
- **Final:** Every 1000 steps
- **Reason:** More frequent monitoring and early stopping detection
- **Impact:** Better tracking of model performance, earlier detection of overfitting

### 3. Batch Size Optimization
- **Initial:** 3 per device
- **Intermediate:** 2 per device
- **Final:** 1 per device (for `train_working.py`), 2 per device (for `train_custom_dataset.py`)
- **Reason:** Resolve CUDA OOM errors
- **Impact:** Lower memory usage, stable training

### 4. Early Stopping Implementation
- **Added:** `EarlyStoppingCallback` with patience=1
- **Reason:** Stop training when validation loss stops improving
- **Impact:** Saves time and compute resources

### 5. Library Upgrades
- **Transformers:** 4.41.2 → 4.45.0
- **Accelerate:** 0.31.0 → 1.12.0
- **Reason:** Fix compatibility issues, especially optimizer errors
- **Impact:** Resolved `AttributeError: 'AdamW' object has no attribute 'train'`

### 6. Optimizer Changes
- **Initial:** `adamw_8bit`
- **Attempted:** `paged_adamw_8bit`
- **Final:** `adamw_torch`
- **Reason:** Compatibility issues with accelerate 0.31.0
- **Impact:** Stable training without optimizer errors

### 7. Data Loading Optimizations
- **Dataloader workers:** 0 → 12 → 4 (final)
- **Prefetch factor:** 0 → 4 → 2 (final)
- **Pin memory:** False → True
- **Reason:** Balance between speed and memory usage
- **Impact:** Faster data loading without excessive memory overhead

### 8. Evaluation Safety Settings
- **Eval batch size:** 4 → 2 → 1
- **Eval accumulation steps:** 8 → 16
- **Reason:** Prevent OOM during evaluation
- **Impact:** Stable evaluation runs

---

## File Changes

### `scripts/train_working.py`

#### Model Loading
- Added proper device mapping for multi-GPU DDP
- Set `torch.cuda.set_device(local_rank)` for DDP processes
- Added `model.config.use_cache = False` for training

#### LoRA Configuration
- Updated target_modules to include `gate_up_proj` (2025 canonical list)
- Final list: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj"]`

#### Training Arguments
```python
# Key settings:
- max_seq_length: 1026 (default 2048)
- per_device_train_batch_size: 1
- per_device_eval_batch_size: 1
- gradient_accumulation_steps: 9
- eval_steps: 1000 (was 5000)
- save_steps: 1000 (was 5000)
- eval_strategy: "steps" (updated from deprecated evaluation_strategy)
- eval_accumulation_steps: 16
- dataloader_num_workers: 4
- dataloader_prefetch_factor: 2
- optim: "adamw_torch" (was "adamw_8bit")
- early_stopping_patience: 1
```

#### Early Stopping
- Added `EarlyStoppingCallback` import
- Implemented callback with patience=1, threshold=0.0
- Only enabled when validation dataset is provided

### `phi3_lora_training/start_training_fixed.sh`

#### Configuration Display
- Updated sequence length display: 1026
- Updated batch size display: 1
- Effective batch size: 18 (1 × 9 × 2 GPUs)

#### Launch Parameters
```bash
--max_seq_length 1026
--batch_size 1
--gradient_accumulation_steps 9
--num_epochs 3
--learning_rate 2e-4
--lora_r 64
--lora_alpha 16
```

### `phi3_lora_training/requirements.txt`
- No changes (already had correct dependencies)

---

## Issues Resolved

### 1. Multi-GPU Device Mapping Error
**Error:** `ValueError: You can't train a model that has been loaded in 8-bit precision on a different device...`

**Fix:** 
- Set `torch.cuda.set_device(local_rank)` before model loading
- Use `device_map=None` for DDP, let DDP handle placement

### 2. Optimizer Compatibility Error
**Error:** `AttributeError: 'AdamW' object has no attribute 'train'`

**Fix:**
- Upgraded accelerate from 0.31.0 to 1.12.0
- Changed optimizer from `adamw_8bit` to `adamw_torch`

### 3. CUDA Out of Memory
**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Fixes Applied:**
- Reduced batch size: 3 → 2 → 1
- Reduced sequence length: 2048 → 1536 → 1026
- Reduced dataloader workers: 12 → 4
- Reduced prefetch factor: 4 → 2
- Reduced eval batch size: 4 → 2 → 1
- Increased eval accumulation steps: 8 → 16
- Killed VLLM processes using GPU memory

### 4. Deprecation Warnings
**Warning:** `evaluation_strategy is deprecated`

**Fix:** Changed to `eval_strategy` in TrainingArguments

### 5. torch.compile() Incompatibility
**Issue:** torch.compile() doesn't work with quantized models + PEFT

**Fix:** Removed torch.compile() calls (commented out)

---

## Current Training Configuration

### Active Training (train_custom_dataset.py)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python scripts/train_custom_dataset.py \
    --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
    --train_file scripts/datasets/merged_train_english_only_short.jsonl \
    --val_file scripts/datasets/merged_val_english_only_short.jsonl \
    --output_dir /home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2 \
    --max_seq_length 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 9 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --save_steps 1000 \
    --eval_steps 1000
```

**Key Differences from train_working.py:**
- Using `train_custom_dataset.py` instead of `train_working.py`
- Single GPU (CUDA_VISIBLE_DEVICES=0) instead of multi-GPU
- Sequence length: 1024 vs 1026
- Batch size: 2 vs 1
- Using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for memory management
- Using shorter dataset files (`*_short.jsonl`)

### train_working.py Configuration (Multi-GPU)
- Sequence length: 1026
- Batch size: 1 per device
- Effective batch: 18 (1 × 9 × 2 GPUs)
- Eval frequency: Every 1000 steps
- Save frequency: Every 1000 steps
- Early stopping: Enabled (patience=1)

---

## Performance Optimizations Applied

1. **Memory Optimizations:**
   - Reduced sequence length (33% reduction)
   - Reduced batch size (50-67% reduction)
   - Reduced dataloader workers and prefetch
   - Increased eval accumulation steps

2. **Speed Optimizations:**
   - Enabled dataloader pin memory
   - Optimized dataloader workers (balanced)
   - Reduced sequence length for faster processing

3. **Stability Optimizations:**
   - Upgraded libraries for compatibility
   - Fixed device mapping for multi-GPU
   - Added early stopping to prevent overfitting
   - Increased eval accumulation to prevent OOM

---

## Environment Variables

### Recommended for Training
```bash
# Memory management (helps with fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Single GPU training (if needed)
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU training (uses all available)
# No CUDA_VISIBLE_DEVICES needed, torchrun handles it
```

---

## Dataset Information

### Training Dataset
- **File:** `merged_train_english_only.jsonl` (or `*_short.jsonl`)
- **Size:** ~366,765 samples (full), varies for short version
- **Language:** English-only (filtered)
- **Format:** JSONL with messages format

### Validation Dataset
- **File:** `merged_val_english_only.jsonl` (or `*_short.jsonl`)
- **Size:** ~40,760 samples (full), varies for short version
- **Language:** English-only (filtered)

---

## Expected Training Metrics

### With Current Configuration (train_custom_dataset.py)
- **Sequence length:** 1024 tokens
- **Batch size:** 2 per device
- **Gradient accumulation:** 9 steps
- **Effective batch size:** 18 (2 × 9 × 1 GPU)
- **Epochs:** 3
- **Evaluation:** Every 1000 steps
- **Checkpoint saving:** Every 1000 steps

### Training Steps Calculation
- Steps per epoch: ~(dataset_size / effective_batch_size)
- Total steps: steps_per_epoch × 3 epochs
- Evaluation runs: At steps 1000, 2000, 3000, etc.

---

## Notes

1. **Two Training Scripts:**
   - `train_working.py`: Optimized for multi-GPU with DDP
   - `train_custom_dataset.py`: More flexible, currently in use

2. **Memory Management:**
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with memory fragmentation
   - Single GPU training avoids DDP overhead

3. **Early Stopping:**
   - Only works if validation dataset is provided
   - Stops if validation loss doesn't improve for 1 evaluation cycle

4. **Checkpoint Management:**
   - `save_total_limit=3` keeps only last 3 checkpoints
   - Best model loaded at end if validation enabled

---

## Future Optimization Opportunities

1. **If memory allows:**
   - Increase batch size back to 2-3
   - Increase sequence length to 1536
   - Increase dataloader workers to 8-12

2. **If speed is priority:**
   - Reduce gradient accumulation (but maintain effective batch size)
   - Increase sequence length (if memory allows)
   - Use torch.compile() if quantized + PEFT compatibility improves

3. **If quality is priority:**
   - Increase sequence length to 2048
   - Increase LoRA rank (r=128)
   - Train for more epochs

---

## Version Information

- **Transformers:** 4.45.0
- **Accelerate:** 1.12.0
- **PyTorch:** 2.9.1
- **PEFT:** 0.11.1
- **BitsAndBytes:** >=0.40.0

---

## Related Documentation

- **Evaluation Process:** See `EVALUATION_PROCESS.md` for detailed evaluation setup and dataset information
- **Training Setup:** See `README_TRAINING_SETUP.md` for initial setup instructions

---

**Last Updated:** December 9, 2025

