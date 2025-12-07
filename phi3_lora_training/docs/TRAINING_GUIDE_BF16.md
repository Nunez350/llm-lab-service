# Phi-3 Medium Training Guide - Full bf16 Configuration

Complete guide for training Phi-3 Medium (14B) with full bf16 precision, optimized for RTX 5090 (32GB VRAM).

## Overview

This guide documents the **full bf16 precision training** setup for Phi-3 Medium, which provides:
- **Faster training** than 8-bit quantization
- **Better accuracy** than quantized models
- **Optimized for high-end GPUs** (RTX 5090 with 32GB VRAM)

### Key Features

- ✅ Full bf16 precision (no quantization)
- ✅ Fused optimizer (`adamw_torch_fused`)
- ✅ Gradient checkpointing (memory optimization)
- ✅ Eager attention (Phi-3 compatible)
- ✅ Optimized dataloader
- ✅ Conditional bitsandbytes import (prevents initialization errors)

## Current Working Configuration

**Memory-optimized settings** (for 32GB GPU):

```bash
--max_seq_length 1536          # Reduced from 2048 to fit in memory
--batch_size 1                 # Smallest batch to avoid OOM
--gradient_accumulation_steps 64 # Effective batch size: 64
--num_epochs 3
--learning_rate 2e-4
--lora_r 64
--lora_alpha 16
--eval_steps 5000
```

**Model settings:**
- Precision: Full bf16 (no quantization)
- Optimizer: `adamw_torch_fused` (auto-selected)
- Attention: Eager (Phi-3 requirement, Flash Attention disabled)
- Gradient checkpointing: Enabled

## Time Estimates

- **Current (max_seq_length=1536)**: ~93 hours (3.9 days)
- **With max_seq_length=1024**: ~54 hours (2.2 days) - **Saves 39 hours (42% faster)**

## Key Code Changes

1. **Conditional bitsandbytes import** - Only imports when `--use_8bit` is used
2. **Flash Attention disabled** - Always uses eager attention for Phi-3
3. **DynamicCache compatibility patch** - Prevents compatibility errors

See `REPRODUCTION_GUIDE.md` for complete step-by-step instructions.
