#!/bin/bash
# Start training script for Phi-3 Medium fine-tuning
# Uses train_working.py which prevents OOM errors with eval_accumulation_steps
# Uses labvenv virtual environment with stable library versions

set -e  # Exit on error

echo "=================================================="
echo "Starting Phi-3 Medium Training (Memory Optimized)"
echo "=================================================="
echo ""

# Change to phi3_lora_training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if labvenv exists in parent directory
if [ ! -d "../labvenv" ]; then
    echo "Error: labvenv not found in parent directory!"
    echo "Run ./setup_training_env.sh first"
    exit 1
fi
# Multi-GPU config
# MULTI_GPU=0 → single GPU (default)
# MULTI_GPU=1 → use torchrun with NUM_GPUS processes
MULTI_GPU=${MULTI_GPU:-0}
NUM_GPUS=${NUM_GPUS:-2}   # default to 2 GPUs when MULTI_GPU=1

# Activate virtual environment
echo "[1/3] Activating labvenv..."
source ../labvenv/bin/activate
echo "  ✓ Virtual environment activated"

# Verify critical versions
TRANSFORMERS_VERSION=$(pip show transformers 2>/dev/null | grep "Version:" | awk '{print $2}')
if [ "$TRANSFORMERS_VERSION" != "4.41.2" ]; then
    echo "  ✗ ERROR: Wrong transformers version ($TRANSFORMERS_VERSION)"
    echo "    Expected 4.41.2. Training will crash at step 500!"
    echo "    Run ./setup_training_env.sh to fix"
    exit 1
fi

# Check if output directory exists, create if not
OUTPUT_DIR="/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "[2/3] Creating output directory..."
    mkdir -p "$OUTPUT_DIR"
    echo "  ✓ Created $OUTPUT_DIR"
fi

# Kill any existing training processes
echo ""
echo "[3/3] Checking for existing training processes..."
EXISTING_PROCESSES=$(ps aux | grep -E "python.*train_" | grep -v grep | wc -l)
if [ "$EXISTING_PROCESSES" -gt 0 ]; then
    echo "  WARNING: Found $EXISTING_PROCESSES existing training process(es)"
    echo "  Kill them? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        pkill -9 -f "python.*train_" || true
        sleep 2
        echo "  ✓ Killed existing processes"
    fi
fi

# Display training configuration
echo ""
echo "=================================================="
echo "Training Configuration"
echo "=================================================="
echo "Script: train_working.py (memory-optimized with eval_accumulation_steps)"
echo "Model: Phi-3-Medium (14B parameters)"
echo "Training data: scripts/datasets/merged_train_english_only.jsonl (English-only, 366,749 sequences)"
echo "Validation data: scripts/datasets/merged_val_english_only.jsonl (English-only, 40,755 sequences)"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Hyperparameters:"
echo "  - Max sequence length: 2048"
echo "  - Batch size: 3 (optimized for RTX 5090)"
echo "  - Gradient accumulation: 6 (effective batch size: 18)"
echo "  - Epochs: 3"
echo "  - Learning rate: 2e-4"
echo "  - LoRA rank: 64"
echo "  - LoRA alpha: 16"
echo ""
echo "Memory Optimizations:"
echo "  - Quantization: 8-bit (prevents OOM during eval)"
echo "  - Precision: bfloat16"
echo "  - Eval accumulation steps: 8 (KEY: prevents OOM during validation)"
echo "  - Eval batch size: 4"
echo "  - torch.compile: enabled (max-autotune for +30-40% speed)"
echo "  - Optimizer: adamw_torch (faster & more stable)"
echo ""
echo "Checkpoints saved every 2000 steps"
echo "Evaluation runs every 2000 steps"
echo "Best model loaded at end based on eval_loss"
echo "=================================================="
echo ""
echo "Press Enter to start training, or Ctrl+C to cancel..."
read -r

# Use train_working.py from main scripts directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../scripts/train_working.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "✗ ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# Start training in background
echo ""
echo "Starting training in background..."
# Decide which GPUs to use
# If MULTI_GPU=1 and NUM_GPUS=2, default to GPUs 0,1 unless CUDA_VISIBLE_DEVICES is already set
if [ "$MULTI_GPU" = "1" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # Default to first NUM_GPUS GPUs (here we assume 0,1 for NUM_GPUS=2)
        CUDA_VISIBLE_DEVICES="0,1"
    fi
    echo "Using MULTI_GPU=1 with NUM_GPUS=$NUM_GPUS on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    # Single-GPU mode, default to GPU 0 if not set
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES="0"
    fi
    echo "Using single-GPU mode on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
export CUDA_VISIBLE_DEVICES

# Build the launch command (python vs torchrun)
if [ "$MULTI_GPU" = "1" ]; then
    # True data-parallel training via torchrun
    LAUNCH_CMD=(torchrun --nproc_per_node="$NUM_GPUS" "$TRAIN_SCRIPT")
else
    # Classic single-process training
    LAUNCH_CMD=(python "$TRAIN_SCRIPT")
fi

echo ""
echo "Starting training in background..."
(
  "${LAUNCH_CMD[@]}" \
    --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
    --train_file "$SCRIPT_DIR/../scripts/datasets/merged_train_english_only.jsonl" \
    --val_file "$SCRIPT_DIR/../scripts/datasets/merged_val_english_only.jsonl" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 1536 \
    --batch_size 1 \
    --gradient_accumulation_steps 18 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    > "$OUTPUT_DIR/training.log" 2>&1 &
)

# Give it a moment to start
sleep 2

# Get PID of training process (works for both python and torchrun)
TRAIN_PID=$(pgrep -f "$TRAIN_SCRIPT" | head -n 1)

if [ -z "$TRAIN_PID" ]; then
    echo "✗ ERROR: Training process failed to start!"
    echo "Check $OUTPUT_DIR/training.log for errors"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Training started successfully!"
echo "=================================================="
echo ""
echo "Process ID: $TRAIN_PID"
echo "Log file: $OUTPUT_DIR/training.log"
echo ""
echo "Monitor training:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "View current progress:"
echo "  watch -n 10 \"tail -5 $OUTPUT_DIR/training.log\""
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""


# Get PID of training process
sleep 2
TRAIN_PID=$(ps aux | grep -E "python.*train_working" | grep -v grep | awk '{print $2}')
...

# Get PID of training process
sleep 2
TRAIN_PID=$(ps aux | grep -E "python.*train_working" | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PID" ]; then
    echo "✗ ERROR: Training process failed to start!"
    echo "Check $OUTPUT_DIR/training.log for errors"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Training started successfully!"
echo "=================================================="
echo ""
echo "Process ID: $TRAIN_PID"
echo "Log file: $OUTPUT_DIR/training.log"
echo ""
echo "Monitor training:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "View current progress:"
echo "  watch -n 10 \"tail -5 $OUTPUT_DIR/training.log\""
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
