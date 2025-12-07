# Quick Start Guide - Phi-3 Medium Training

Get training running in under 5 minutes!

## Prerequisites

- NVIDIA GPU with 20GB+ VRAM
- Python 3.8+
- CUDA toolkit installed
- Phi-3 Medium model downloaded to `/home/rnu/mnt/models/phi_models/phi3-medium/`
- Dataset files in `scripts/datasets/`

## 3-Step Quick Start

### Step 1: Setup Environment (One-time)

```bash
cd /home/rnu/mnt/models/llm-lab-service
./setup_training_env.sh
```

**What this does:**
- Creates `labvenv` virtual environment
- Installs transformers 4.41.2, accelerate 0.31.0, peft 0.11.1
- Verifies all versions are correct

**Time:** ~5-10 minutes

### Step 2: Start Training

```bash
./start_training.sh
```

**What this does:**
- Activates `labvenv`
- Verifies environment is correct
- Starts training in background
- Creates log file

**Time:** Immediate

### Step 3: Monitor Training

```bash
# Watch training progress
tail -f /home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log

# Or use watch command for periodic updates
watch -n 10 "tail -5 /home/rnu/mnt/models/unsloth/phi3-finetuned-stable/training.log"
```

---

## That's It!

Training will run for approximately **65 hours** for 3 epochs.

### What to Expect

**After 10 minutes:**
- Model loaded
- Dataset formatted
- Training started (step 1-5)

**After 1 hour:**
- ~500-600 steps completed
- First checkpoint saved
- First evaluation complete
- Loss should be ~0.9-1.0

**After 65 hours:**
- Training complete
- Model saved to `/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/`
- Ready to use!

### Monitor GPU

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Or use watch
watch -n 1 nvidia-smi
```

### Stop Training

```bash
# Find process ID
ps aux | grep train_custom_dataset

# Kill it
kill <PID>

# Or kill all training processes
pkill -9 -f train_custom_dataset.py
```

---

## Need More Details?

See `README_TRAINING_SETUP.md` for:
- Detailed explanation of each step
- Troubleshooting guide
- Parameter tuning
- Evaluation instructions
- How to use the trained model

---

## Common Issues

### "DynamicCache error at step 500"

**Fix:** Verify transformers version:
```bash
source labvenv/bin/activate
pip show transformers  # Should be 4.41.2
```

If wrong version, run `./setup_training_env.sh` again.

### "Out of memory"

**Fix:** Reduce batch size in `start_training.sh`:
```bash
# Change this line:
--batch_size 2 \

# To:
--batch_size 1 \
```

### "No module named 'transformers'"

**Fix:** Activate virtual environment:
```bash
source labvenv/bin/activate
```

---

## Files Created

```
/home/rnu/mnt/models/llm-lab-service/
├── labvenv/                          # Virtual environment
├── setup_training_env.sh             # Environment setup
├── start_training.sh                 # Start training
└── requirements.txt                   # Dependencies

/home/rnu/mnt/models/unsloth/phi3-finetuned-stable/
├── training.log                      # Training logs
├── checkpoint-500/                   # Checkpoints
├── checkpoint-1000/
└── ...                               # Final model
```

---

## Success Indicators

✅ Training is working if you see:
```
{'loss': 1.0028, 'grad_norm': 0.089, 'learning_rate': 0.00019989, 'epoch': 0.0}
{'loss': 0.9342, 'grad_norm': 0.094, 'learning_rate': 0.00019979, 'epoch': 0.01}
```

✅ Passed the critical step 500 without errors

✅ Checkpoints being saved regularly

❌ Training failed if:
- DynamicCache error at step 500
- Process exits unexpectedly
- GPU memory errors

---

Ready to train? Run:
```bash
./setup_training_env.sh && ./start_training.sh
```
