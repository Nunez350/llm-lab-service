# Phi-3 Medium Fine-Tuning Setup Guide

This guide provides complete steps to reproduce the successful training setup for Phi-3 Medium model using stable library versions.

## Problem Solved

Previous training attempts crashed at step 500 with this error:
```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'
```

This was caused by incompatible library versions between newer transformers (4.57.x) and the Phi-3 model code.

## Solution Overview

1. **Separate virtual environment** (`labvenv`) with stable library versions
2. **Downgraded transformers** to 4.41.2 (from 4.57.x)
3. **Compatible PEFT** version 0.11.1 (from 0.18.x)
4. **Disabled model cache** to bypass DynamicCache compatibility issues
5. **Keep evaluation during training** - now works without crashing

---

## Quick Start

```bash
# 1. Run automated setup
./setup_training_env.sh

# 2. Start training
./start_training.sh
```

---

## Manual Setup Instructions

### Step 1: Create Virtual Environment

```bash
cd /home/rnu/mnt/models/llm-lab-service
python3 -m venv labvenv
```

### Step 2: Activate and Install Dependencies

```bash
source labvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Critical versions installed:**
- `transformers==4.41.2` (NOT 4.57.x - causes DynamicCache errors)
- `accelerate==0.31.0` (NOT 1.12.x - incompatible with newer PEFT)
- `peft==0.11.1` (NOT 0.18.x - incompatible with accelerate 0.31.0)

### Step 3: Verify train_custom_dataset.py Has the Fix

The file `../scripts/train_custom_dataset.py` (consolidated script) must include this line after model preparation (around line 113):

```python
# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Disable cache to avoid DynamicCache compatibility issues during training and eval
model.config.use_cache = False
```

This is **critical** - it prevents the DynamicCache error during evaluation.

### Step 4: Prepare Your Dataset

Ensure you have your dataset files:
- `scripts/datasets/merged_train.jsonl` - Training data
- `scripts/datasets/merged_val.jsonl` - Validation data

Format should be JSONL with messages structure:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Step 5: Start Training

**Option A: Interactive (see output in terminal)**
```bash
cd /home/rnu/mnt/models/llm-lab-service
source labvenv/bin/activate

CUDA_VISIBLE_DEVICES=0 python ../scripts/train_custom_dataset.py \
  --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
  --train_file ../scripts/datasets/merged_train.jsonl \
  --val_file ../scripts/datasets/merged_val.jsonl \
  --output_dir /home/rnu/mnt/models/unsloth/phi3-finetuned-stable \
  --max_seq_length 2048 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 64 \
  --lora_alpha 16 \
  --gpu_id 0
```

**Option B: Background (recommended for long training)**
```bash
./start_training.sh
```

### Step 6: Monitor Training

```bash
# Watch training progress
tail -f /home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log

# Check GPU usage
nvidia-smi

# Verify process is running
ps aux | grep train_custom_dataset

# Check current progress (every 10 seconds)
watch -n 10 "tail -5 /home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log"
```

---

## Training Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_seq_length` | 2048 | Maximum input sequence length |
| `batch_size` | 2 | Per-device batch size (GPU memory constraint) |
| `gradient_accumulation_steps` | 4 | Effective batch size = 2 × 4 = 8 |
| `num_epochs` | 3 | Number of training passes |
| `learning_rate` | 2e-4 | Standard for LoRA fine-tuning |
| `lora_r` | 64 | LoRA rank (higher = more parameters) |
| `lora_alpha` | 16 | LoRA scaling factor |

**Effective batch size:** 8 (batch_size × gradient_accumulation_steps)
**Trainable parameters:** ~85M (0.61% of 14B total)

---

## Expected Training Behavior

### Normal Loss Values
- **Initial loss:** 1.0 - 1.5
- **During training:** 0.85 - 1.10 (fluctuating but trending down)
- **Final loss:** 0.7 - 0.9

### Training Speed
- **~1.7 seconds per step** (on RTX 4090 or similar)
- **Total steps:** ~139,632 (depends on dataset size)
- **Estimated time:** ~65 hours for 3 epochs

### Checkpoints
Saved every 500 steps to:
```
/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── ...
```

### Evaluation Metrics
Runs every 500 steps, showing validation loss:
```
{'loss': 0.95, 'eval_loss': 0.89, 'epoch': 0.5}
```

---

## Troubleshooting

### Issue: DynamicCache error at step 500

**Symptoms:**
```
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'
```

**Solution:**
1. Verify you're using `labvenv` (not unsloth venv)
2. Check transformers version: `pip show transformers` (must be 4.41.2)
3. Ensure `model.config.use_cache = False` is in `../scripts/train_custom_dataset.py` (consolidated script)

### Issue: PEFT import error

**Symptoms:**
```
ImportError: cannot import name 'clear_device_cache' from 'accelerate.utils.memory'
```

**Solution:**
```bash
source labvenv/bin/activate
pip uninstall -y peft
pip install peft==0.11.1
```

### Issue: Out of memory

**Solution:**
Reduce batch size or sequence length:
```bash
python ../scripts/train_custom_dataset.py \
  --batch_size 1 \
  --max_seq_length 1024 \
  ...
```

### Issue: Training too slow

**Solutions:**
1. Reduce `max_seq_length` to 1024 or 512
2. Increase `batch_size` if you have GPU memory
3. Use gradient checkpointing (already enabled)

### Issue: Multiple processes running

**Kill all training processes:**
```bash
pkill -9 -f train_custom_dataset.py
```

---

## File Structure

```
llm-lab-service/
├── phi3_lora_training/
│   ├── README_TRAINING_SETUP.md      # This file
│   ├── requirements.txt              # Python dependencies
│   ├── setup_training_env.sh         # Automated environment setup
│   └── start_training.sh             # Start training script
├── labvenv/                          # Virtual environment (created by setup)
└── scripts/
    ├── train_custom_dataset.py       # Consolidated training script (with fix)
    ├── datasets/
    │   ├── merged_train.jsonl        # Training data
    │   └── merged_val.jsonl          # Validation data
    └── evaluate_model.py             # Evaluation script (optional)
```

---

## After Training Completes

### 1. Find Your Trained Model

The final model will be saved to:
```
/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/
```

### 2. Test the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/rnu/mnt/models/phi_models/phi3-medium/",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "/home/rnu/mnt/models/phi_models/phi3-medium/"
)

# Generate
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### 3. Evaluate Performance

Run the evaluation framework:
```bash
cd scripts
./run_evaluation.sh /home/rnu/mnt/models/unsloth/phi3-finetuned-stable
```

---

## Key Differences from Previous Attempts

| Aspect | Previous (Failed) | Current (Working) |
|--------|------------------|-------------------|
| Environment | unsloth venv | labvenv (isolated) |
| transformers | 4.57.2 | 4.41.2 |
| accelerate | 1.12.0 | 0.31.0 |
| peft | 0.18.0 | 0.11.1 |
| Model cache | Enabled | **Disabled** |
| Crashes at step 500 | ✗ YES | ✓ NO |

---

## Technical Details

### Why transformers 4.41.2?

Transformers 4.57.x introduced breaking changes to the `DynamicCache` class:
- Removed `get_usable_length()` method
- Added `get_seq_length()` method instead
- Phi-3 model code still uses old API
- Version 4.41.2 is the last stable version before these changes

### Why model.config.use_cache = False?

- Disables the KV cache during training and evaluation
- Bypasses the DynamicCache compatibility issue entirely
- Slightly slower but prevents crashes
- Essential for evaluation to work with stable transformers

### Why separate virtual environment?

- Prevents conflicts with unsloth's newer library versions
- Allows reproducible setup
- Easy to tear down and recreate
- Isolates dependencies

---

## Resources

- **Training logs:** `/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log`
- **Checkpoints:** `/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/checkpoint-*/`
- **Evaluation guide:** `EVALUATION_GUIDE.md`
- **Phi-3 documentation:** https://huggingface.co/microsoft/Phi-3-medium-4k-instruct

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify library versions: `pip list | grep -E "transformers|accelerate|peft"`
3. Review training logs for error messages
4. Ensure GPU has sufficient memory (14B model needs ~20GB with 4-bit quantization)

---

## License

This setup is compatible with the Phi-3 model license (MIT).
