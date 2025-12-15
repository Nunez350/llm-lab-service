# Training Run Log

## December 9, 2025 - Training Execution

### Actual Command Used

This is the exact command that was executed to start training on December 9, 2025:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_custom_dataset.py \
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
    > training_resume.log 2>&1 &
```

### Configuration Summary

- **Date**: December 9, 2025
- **Model**: Phi-3 Medium (14B parameters)
- **Training Dataset**: `merged_train_english_only_short.jsonl` (355,896 sequences, ≤1024 tokens)
- **Validation Dataset**: `merged_val_english_only_short.jsonl` (39,607 sequences, ≤1024 tokens)
- **Max Sequence Length**: 1024 tokens
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 9 steps
- **Effective Batch Size**: 18 (2 × 9)
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Save Steps**: 1000
- **Eval Steps**: 1000
- **Early Stopping Patience**: 3 evaluations
- **Resume**: Auto (resumes from latest checkpoint if available)
- **Output Directory**: `/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2`
- **Log File**: `training_resume.log`

### Dataset Details

**Short Sequences (used for this training):**
- Training: 355,896 sequences (≤1024 tokens) - 97.0% of total
- Validation: 39,607 sequences (≤1024 tokens) - 97.2% of total

**Long Sequences (saved for later):**
- Training: 10,869 sequences (>1024 tokens) - 3.0% of total
- Validation: 1,153 sequences (>1024 tokens) - 2.8% of total

### Optimizations Applied

1. **Dataset Splitting**: Sequences split by length using `split_by_length.py`
2. **Sequence Length Reduction**: Reduced from 1536 to 1024 tokens (~42% speedup)
3. **Short Sequences Only**: Using 97% of data that fits in 1024 tokens
4. **Frequent Checkpoints**: Saving every 1000 steps for safety
5. **Frequent Evaluation**: Evaluating every 1000 steps
6. **Early Stopping**: Stops if no improvement after 3 evaluations

### Notes

- Training was started with `nohup` to run in background
- Logs are written to `training_resume.log`
- Uses `train_custom_dataset.py` script (not `train_working.py`)
- Batch size is 2 (not 1 as in some other configurations)
- Auto-resume enabled to continue from checkpoints if training is interrupted

