#!/bin/bash

# Run evaluation comparing fine-tuned vs base model
# Uses two GPUs: GPU 0 for fine-tuned, GPU 1 for base

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=================================================="
echo "Model Evaluation - Two GPU Mode"
echo "=================================================="
echo ""

# Check env
if [ ! -d "labvenv" ]; then
    echo "Error: labvenv not found! Run setup_training_env.sh first."
    exit 1
fi

echo "[1/2] Activating labvenv..."
source labvenv/bin/activate
echo "  Virtual environment activated"

# Default values
MODEL_PATH="${MODEL_PATH:-/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/checkpoint-22000}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/rnu/mnt/models/phi_models/phi3-medium/}"
VAL_FILE="${VAL_FILE:-$SCRIPT_DIR/../scripts/datasets/merged_val_english_only_short.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-v2/evaluation}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

echo ""
echo "=================================================="
echo "Evaluation Configuration"
echo "=================================================="
echo "Fine-tuned Model: $MODEL_PATH"
echo "Base Model:       $BASE_MODEL_PATH"
echo "Validation File:  $VAL_FILE"
echo "Output Dir:       $OUTPUT_DIR"
echo "Num Samples:      $NUM_SAMPLES"
echo "=================================================="
echo ""

# Verify files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "ERROR: Base model path not found: $BASE_MODEL_PATH"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: Validation file not found: $VAL_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[2/2] Starting evaluation..."
echo ""

python "$SCRIPT_DIR/evaluate_model.py" \
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES"

echo ""
echo "=================================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
