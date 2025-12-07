#!/bin/bash
# Dataset preparation script for Phi-3 training
# Downloads and prepares OpenHermes 2.5 + SlimOrca datasets

set -e  # Exit on error

echo "=================================================="
echo "Dataset Preparation for Phi-3 Training"
echo "=================================================="
echo ""

# Change to phi3_lora_training directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if labvenv exists
if [ ! -d "../labvenv" ]; then
    echo "Warning: labvenv not found in parent directory!"
    echo "Run ./setup_training_env.sh first to create virtual environment."
    echo ""
    echo "Continue anyway with system Python? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted. Please run ./setup_training_env.sh first."
        exit 1
    fi
else
    echo "[1/3] Activating labvenv..."
    source ../labvenv/bin/activate
    echo "  ✓ Virtual environment activated"
fi

# Check disk space
echo ""
echo "[2/3] Checking disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 10 ]; then
    echo "  ⚠ Warning: Low disk space (${AVAILABLE_GB}GB available)"
    echo "  Recommended: At least 10GB free"
    echo "  Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted. Please free up disk space."
        exit 1
    fi
else
    echo "  ✓ Sufficient disk space (${AVAILABLE_GB}GB available)"
fi

# Display preparation info
echo ""
echo "=================================================="
echo "Dataset Preparation Plan"
echo "=================================================="
echo ""
echo "Sources:"
echo "  - OpenHermes 2.5 (teknium/OpenHermes-2.5)"
echo "  - SlimOrca (Open-Orca/SlimOrca)"
echo ""
echo "Processing:"
echo "  1. Download datasets from HuggingFace"
echo "  2. Normalize conversation formats"
echo "  3. Apply quality filters"
echo "  4. Remove duplicates"
echo "  5. Split into train/validation (90/10)"
echo ""
echo "Output:"
echo "  - scripts/datasets/merged_train.jsonl (~372k examples)"
echo "  - scripts/datasets/merged_val.jsonl (~41k examples)"
echo "  - scripts/datasets/dataset_stats.json (metrics)"
echo ""
echo "Estimated time: 30-45 minutes"
echo "Disk space needed: ~5GB during processing, ~500MB final"
echo "=================================================="
echo ""
echo "Press Enter to start, or Ctrl+C to cancel..."
read -r

# Change to scripts directory within this package
cd scripts

# Run dataset preparation
echo ""
echo "[3/3] Running dataset preparation..."
echo ""
python prepare_dataset.py

# Check if files were created
echo ""
echo "=================================================="
echo "Verifying Output Files"
echo "=================================================="
echo ""

if [ ! -f "datasets/merged_train.jsonl" ]; then
    echo "✗ ERROR: merged_train.jsonl not found!"
    exit 1
fi

if [ ! -f "datasets/merged_val.jsonl" ]; then
    echo "✗ ERROR: merged_val.jsonl not found!"
    exit 1
fi

if [ ! -f "datasets/dataset_stats.json" ]; then
    echo "✗ ERROR: dataset_stats.json not found!"
    exit 1
fi

# Show file sizes
echo "✓ All files created successfully!"
echo ""
echo "File sizes:"
ls -lh datasets/merged_train.jsonl | awk '{print "  " $9 ": " $5}'
ls -lh datasets/merged_val.jsonl | awk '{print "  " $9 ": " $5}'
ls -lh datasets/dataset_stats.json | awk '{print "  " $9 ": " $5}'

# Count examples
TRAIN_COUNT=$(wc -l < datasets/merged_train.jsonl)
VAL_COUNT=$(wc -l < datasets/merged_val.jsonl)

echo ""
echo "Example counts:"
echo "  Training: $(printf "%'d" $TRAIN_COUNT) examples"
echo "  Validation: $(printf "%'d" $VAL_COUNT) examples"
echo "  Total: $(printf "%'d" $((TRAIN_COUNT + VAL_COUNT))) examples"

# Show sample statistics
if command -v python &> /dev/null; then
    echo ""
    echo "Dataset statistics:"
    python -m json.tool datasets/dataset_stats.json | grep -A 3 "merged" | head -5
fi

echo ""
echo "=================================================="
echo "✓ Dataset Preparation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review dataset statistics:"
echo "     cat scripts/datasets/dataset_stats.json | python -m json.tool"
echo ""
echo "  2. View sample example:"
echo "     head -1 scripts/datasets/merged_train.jsonl | python -m json.tool"
echo ""
echo "  3. Start training:"
echo "     ./start_training.sh"
echo ""
