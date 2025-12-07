#!/bin/bash
# Start training script for Phi-3 Medium fine-tuning
# Uses labvenv virtual environment with stable library versions

set -e  # Exit on error

echo "=================================================="
echo "Starting Phi-3 Medium Training"
echo "=================================================="
echo ""

# Change to project directory
cd /home/rnu/mnt/models/llm-lab-service

# Check if labvenv exists
if [ ! -d "labvenv" ]; then
    echo "Error: labvenv not found!"
    echo "Run ./setup_training_env.sh first"
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating labvenv..."
source labvenv/bin/activate
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
OUTPUT_DIR="/home/rnu/mnt/models/unsloth/phi3-finetuned-stable"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "[2/3] Creating output directory..."
    mkdir -p "$OUTPUT_DIR"
    echo "  ✓ Created $OUTPUT_DIR"
fi

# Kill any existing training processes
echo ""
echo "[3/3] Checking for existing training processes..."
EXISTING_PROCESSES=$(ps aux | grep -E "python.*train_custom_dataset" | grep -v grep | wc -l)
if [ "$EXISTING_PROCESSES" -gt 0 ]; then
    echo "  WARNING: Found $EXISTING_PROCESSES existing training process(es)"
    echo "  Kill them? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        pkill -9 -f train_custom_dataset.py || true
        sleep 2
        echo "  ✓ Killed existing processes"
    fi
fi

# Display training configuration
echo ""
echo "=================================================="
echo "Training Configuration"
echo "=================================================="
echo "Model: Phi-3-Medium (14B parameters)"
echo "Training data: scripts/datasets/merged_train.jsonl"
echo "Validation data: scripts/datasets/merged_val.jsonl"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Hyperparameters:"
echo "  - Max sequence length: 2048"
echo "  - Batch size: 2"
echo "  - Gradient accumulation: 4 (effective batch size: 8)"
echo "  - Epochs: 3"
echo "  - Learning rate: 2e-4"
echo "  - LoRA rank: 64"
echo "  - LoRA alpha: 16"
echo ""
echo "Expected training time: ~65 hours (depends on GPU)"
echo "Checkpoints saved every 500 steps"
echo "Evaluation runs every 500 steps"
echo "=================================================="
echo ""
echo "Press Enter to start training, or Ctrl+C to cancel..."
read -r

# Change to scripts directory
cd scripts

# Start training in background
echo ""
echo "Starting training in background..."
(CUDA_VISIBLE_DEVICES=0 python train_custom_dataset.py \
  --model_path /home/rnu/mnt/models/phi_models/phi3-medium/ \
  --train_file ./datasets/merged_train.jsonl \
  --val_file ./datasets/merged_val.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 2048 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 64 \
  --lora_alpha 16 \
  --gpu_id 0 \
  > "$OUTPUT_DIR/training.log" 2>&1 &)

# Get PID of training process
sleep 2
TRAIN_PID=$(ps aux | grep -E "python.*train_custom_dataset" | grep -v grep | awk '{print $2}')

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
