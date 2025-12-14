#!/bin/bash

# Start training script for Phi-3 Medium fine-tuning
# Memory-optimized launcher for train_working.py

set -e

echo "Starting Phi-3 Medium Training (Memory Optimized)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check env
if [ ! -d "../labvenv" ]; then
    echo "Error: labvenv not found! Run setup_training_env.sh first."
    exit 1
fi

# Multi-GPU switch
MULTI_GPU=${MULTI_GPU:-0}
NUM_GPUS=${NUM_GPUS:-2}

echo "[1/3] Activating labvenv..."
source ../labvenv/bin/activate
echo "  ✓ Virtual environment activated"

TRANSFORMERS_VERSION=$(pip show transformers 2>/dev/null | grep "Version:" | awk '{print $2}')
# Check for minimum version 4.45.0 (2025 best practice - fixes for 4-bit + PEFT + torch.compile)
if [ -n "$TRANSFORMERS_VERSION" ]; then
    REQUIRED_VERSION="4.45.0"
    if ! printf '%s\n%s\n' "$REQUIRED_VERSION" "$TRANSFORMERS_VERSION" | sort -V -C 2>/dev/null; then
        echo "⚠ WARNING: transformers version $TRANSFORMERS_VERSION is below recommended $REQUIRED_VERSION"
        echo "  torch.compile() with 4-bit QLoRA works best with >=4.45.0"
        echo "  Continuing anyway..."
    fi
fi

OUTPUT_DIR="/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[2/3] Checking for existing training processes..."
EXISTING_PROCESSES=$(pgrep -f "train_working.py" | wc -l)
if [ "$EXISTING_PROCESSES" -gt 0 ]; then
    echo "  WARNING: $EXISTING_PROCESSES existing processes found. Auto-killing..."
    pkill -9 -f "train_working.py" || true
    sleep 2
    echo "  ✓ Killed existing processes"
fi

echo ""
echo "Training Configuration"
echo "Seq length:      1026"
echo "Batch size:      1"
echo "Grad accum:      9"
echo "Effective batch: 18"
echo "Learning rate:   2e-4"
echo "LoRA:            r=64, alpha=16"
echo "Quant:           4-bit NF4 (QLoRA recommended)"
echo ""
echo "MULTI_GPU:       $MULTI_GPU"
echo "NUM_GPUS:        $NUM_GPUS"
echo "Output:          $OUTPUT_DIR"
echo ""

TRAIN_SCRIPT="$SCRIPT_DIR/../scripts/train_working.py"

# Verify training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "✗ ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# Verify dataset files exist
TRAIN_FILE="$SCRIPT_DIR/../scripts/datasets/merged_train_english_only_short.jsonl"
VAL_FILE="$SCRIPT_DIR/../scripts/datasets/merged_val_english_only_short.jsonl"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "✗ ERROR: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "✗ ERROR: Validation file not found: $VAL_FILE"
    exit 1
fi

# GPU selection
if [ "$MULTI_GPU" = "1" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
    echo "Using MULTI_GPU=1 → torchrun ($NUM_GPUS GPUs)"
    LAUNCH_CMD=(torchrun --nproc_per_node="$NUM_GPUS" "$TRAIN_SCRIPT")
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
    echo "Using SINGLE GPU → python"
    LAUNCH_CMD=(python "$TRAIN_SCRIPT")
fi

export CUDA_VISIBLE_DEVICES

# Launch training
echo ""
echo "Starting training..."

(
  "${LAUNCH_CMD[@]}" \
    --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --gradient_accumulation_steps 9 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    > "$OUTPUT_DIR/training.log" 2>&1 &
)

sleep 2

# Get PID - try multiple methods for better reliability
TRAIN_PID=$(pgrep -f "train_working.py" | head -n 1)

# If pgrep didn't work, try ps
if [ -z "$TRAIN_PID" ]; then
    TRAIN_PID=$(ps aux | grep -E "python.*train_working.py" | grep -v grep | awk '{print $2}' | head -n 1)
fi

if [ -z "$TRAIN_PID" ]; then
    echo "✗ ERROR: Training failed to start"
    echo "Check: $OUTPUT_DIR/training.log"
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo ""
        echo "Last 20 lines of log:"
        tail -20 "$OUTPUT_DIR/training.log"
    fi
    exit 1
fi

echo ""
echo "✓ Training started successfully!"
echo "PID: $TRAIN_PID"
echo "Log: $OUTPUT_DIR/training.log"
echo ""
echo "Monitor:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "GPU:"
echo "  nvidia-smi"
echo ""
echo "Kill:"
echo "  kill $TRAIN_PID"

