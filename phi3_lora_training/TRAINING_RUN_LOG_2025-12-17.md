# Training Run Log - December 17, 2025

## Training Execution Summary

### Actual Command Used

This is the exact command that was executed to start training on December 9, 2025:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_custom_dataset.py \
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
    --eval_steps 1000 \
    --early_stopping_patience 3 \
    --resume_from_checkpoint auto \
    > /home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/training_early_stop.log 2>&1 &
```

### Configuration Summary

- **Training Start Date**: December 9, 2025
- **Training Completion Date**: December 9, 2025
- **Model**: Phi-3 Medium (14B parameters)
- **Training Dataset**: `merged_train_english_only_short.jsonl` (355,896 sequences, ≤1024 tokens)
- **Validation Dataset**: `merged_val_english_only_short.jsonl` (39,607 sequences, ≤1024 tokens)
- **Max Sequence Length**: 1024 tokens
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 9 steps
- **Effective Batch Size**: 18 (2 × 9)
- **Epochs**: 3 (planned)
- **Learning Rate**: 2e-4
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Save Steps**: 1000
- **Eval Steps**: 1000
- **Early Stopping Patience**: 3 evaluations
- **Resume**: Auto (resumed from checkpoint-28000)
- **Output Directory**: `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2`
- **Log File**: `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/training_early_stop.log`
- **Memory Optimization**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduces memory fragmentation)

### Dataset Details

**Short Sequences (used for this training):**
- Training: 355,896 sequences (≤1024 tokens) - 97.0% of total
- Validation: 39,607 sequences (≤1024 tokens) - 97.2% of total

**Long Sequences (saved for later):**
- Training: 10,869 sequences (>1024 tokens) - 3.0% of total
- Validation: 1,153 sequences (>1024 tokens) - 2.8% of total

### Training Results

**Training Status:** Completed (manually stopped after step 44,000)

**Final Checkpoints:**
- `checkpoint-42000`: eval_loss = 0.7538163065910339
- `checkpoint-43000`: eval_loss = 0.7538078427314758
- `checkpoint-44000`: eval_loss = 0.7537989616394043 ✓ **BEST**

**Training Progress:**
- Started: Resumed from checkpoint-28000
- Completed: Step 44,000 / 59,316 (74% of planned 3 epochs)
- Total Epochs Completed: 2.23 / 3.00
- Training stopped manually (kill -9) after detecting plateau

**Loss Progression (Steps 38000-44000):**

| Step | Training Loss | Eval Loss | Gap |
|------|--------------|-----------|-----|
| 38000 | 0.8001 | 0.7538 | +0.046 |
| 39000 | 0.6976 | 0.7538 | -0.056 |
| 40000 | 0.7405 | 0.7538 | -0.013 |
| 41000 | 0.7540 | 0.7538 | +0.000 |
| 42000 | 0.7983 | 0.7538 | +0.045 |
| 43000 | 0.7548 | 0.7538 | +0.001 |
| 44000 | 0.8186 | **0.7538** | +0.065 |

**Overfitting Analysis:**
- Training loss at step 44000: 0.8186
- Eval loss at step 44000: 0.7538
- Gap: +0.0648 (training loss higher than eval loss)
- **Conclusion**: No overfitting detected - model generalizes well to validation data
- Validation loss still decreasing at stopping point (0.753825 → 0.753799)

**Best Model:**
- Location: `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-44000`
- Eval Loss: 0.7537989616394043 (lowest across all checkpoints)
- Status: Ready for evaluation and deployment

**Log Files:**
- Main log: `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/training_early_stop.log`
- Previous runs: `training.log`, `training_resume.log`, `training_continue.log`

### Optimizations Applied

1. **Dataset Splitting**: Sequences split by length using `split_by_length.py`
2. **Sequence Length Reduction**: Reduced from 1536 to 1024 tokens (~42% speedup)
3. **Short Sequences Only**: Using 97% of data that fits in 1024 tokens
4. **Frequent Checkpoints**: Saving every 1000 steps for safety
5. **Frequent Evaluation**: Evaluating every 1000 steps
6. **Early Stopping**: Configured to stop if no improvement after 3 evaluations
7. **Gradient Checkpointing**: Enabled for memory efficiency
8. **Memory Fragmentation Prevention**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Technical Details

**Hardware:**
- GPU: Single GPU (CUDA_VISIBLE_DEVICES=0)
- Multi-GPU training disabled due to VLLM process on GPU 1

**Software Versions:**
- Transformers: 4.45.0
- Accelerate: 1.12.0
- PEFT: Latest
- PyTorch: 2.x with bfloat16

**Training Script:**
- Script: `scripts/train_custom_dataset.py`
- Optimizer: `adamw_torch` (standard optimizer, compatible with 4-bit QLoRA)
- Quantization: 4-bit NF4 (QLoRA)
- Precision: bfloat16
- Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate_up_proj"]`

### Notes

- Training was started with `nohup` to run in background
- Logs are written to `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/training_early_stop.log`
- Uses `train_custom_dataset.py` script (not `train_working.py`)
- Batch size is 2 (not 1 as in some other configurations)
- Auto-resume enabled to continue from checkpoints if training is interrupted
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set to reduce CUDA memory fragmentation issues
- Training manually stopped after observing plateau in validation loss
- Model shows healthy generalization (no overfitting)

### Next Steps

- [ ] Run comprehensive evaluation on checkpoint-44000
- [ ] Compare perplexity with base model
- [ ] Generate sample outputs for qualitative assessment
- [ ] Consider training on long sequences (>1024 tokens) if needed
- [ ] Deploy best checkpoint for production use
