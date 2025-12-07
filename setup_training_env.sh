#!/bin/bash
# Automated setup script for Phi-3 training environment
# This script creates a clean virtual environment with stable library versions

set -e  # Exit on error

echo "=================================================="
echo "Phi-3 Training Environment Setup"
echo "=================================================="
echo ""

# Change to project directory
cd /home/rnu/mnt/models/llm-lab-service

# Step 1: Create virtual environment
echo "[1/5] Creating virtual environment 'labvenv'..."
if [ -d "labvenv" ]; then
    echo "  WARNING: labvenv already exists. Remove it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf labvenv
        python3 -m venv labvenv
        echo "  ✓ Created fresh labvenv"
    else
        echo "  ✓ Using existing labvenv"
    fi
else
    python3 -m venv labvenv
    echo "  ✓ Created labvenv"
fi

# Step 2: Activate virtual environment
echo ""
echo "[2/5] Activating virtual environment..."
source labvenv/bin/activate
echo "  ✓ Activated labvenv"

# Step 3: Upgrade pip
echo ""
echo "[3/5] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "  ✓ Pip upgraded to $(pip --version | awk '{print $2}')"

# Step 4: Install dependencies
echo ""
echo "[4/5] Installing dependencies from requirements.txt..."
echo "  This may take 5-10 minutes..."
pip install -r requirements.txt > /tmp/pip_install.log 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ All dependencies installed successfully"
else
    echo "  ✗ Error installing dependencies. Check /tmp/pip_install.log"
    exit 1
fi

# Step 5: Verify critical package versions
echo ""
echo "[5/5] Verifying critical package versions..."

TRANSFORMERS_VERSION=$(pip show transformers 2>/dev/null | grep "Version:" | awk '{print $2}')
ACCELERATE_VERSION=$(pip show accelerate 2>/dev/null | grep "Version:" | awk '{print $2}')
PEFT_VERSION=$(pip show peft 2>/dev/null | grep "Version:" | awk '{print $2}')

echo "  transformers: $TRANSFORMERS_VERSION (expected: 4.41.2)"
echo "  accelerate: $ACCELERATE_VERSION (expected: 0.31.0)"
echo "  peft: $PEFT_VERSION (expected: 0.11.1)"

# Check if versions are correct
if [ "$TRANSFORMERS_VERSION" != "4.41.2" ]; then
    echo "  ✗ WARNING: transformers version mismatch!"
    echo "    Expected 4.41.2, got $TRANSFORMERS_VERSION"
    echo "    This may cause DynamicCache errors at step 500!"
    exit 1
fi

if [ "$PEFT_VERSION" != "0.11.1" ]; then
    echo "  ✗ WARNING: peft version mismatch!"
    echo "    Expected 0.11.1, got $PEFT_VERSION"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Verify train_custom_dataset.py has 'model.config.use_cache = False'"
echo "  2. Run: ./start_training.sh"
echo ""
echo "To activate this environment manually:"
echo "  source labvenv/bin/activate"
echo ""
