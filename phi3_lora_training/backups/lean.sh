#!/bin/bash
# Start training script for Phi-3 Medium fine-tuning
# Memory-optimized launcher for train_working.py

set -e

echo "=================================================="
echo "Starting Phi-3 Medium Training (Memory Optimized)"
echo "=================================================="
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
if [ "$TRANSFORMERS_VERSION" != "4.41.2" ]; then
    echo "✗ ERROR: Wrong transformers version ($TRANSFORMERS_VERSION)"
    echo "Expected: 4.41.2"
    exit 1
fi

OUTPUT_DIR="/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[3/3] Checking for existing training processes..."
EXISTING_PROCESSES=$(pgrep -f train_working.py | wc -l)
if [ "$EXISTING_PROCESSES" -gt 0 ]; then
    echo "  WARNING: $EXISTING_PROCESSES existing processes found."
    echo "  Kill them? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        pkill -9 -f train_working.py || true
        sleep 2
        echo "  ✓ Killed"
    fi
fi

echo ""
echo "=================================================="
echo "Training Configuration"
echo "=================================================="
echo "Seq length:      2048"
echo "Batch size:      2"
echo "Grad accum:
