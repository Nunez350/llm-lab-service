#!/bin/bash
# Complete Evaluation Pipeline Runner
# Usage: ./run_evaluation.sh [checkpoint_path]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_MODEL="${BASE_MODEL_PATH:-/home/rnu/mnt/models/phi_models/phi3-medium}"
VAL_FILE="${VAL_FILE:-./datasets/merged_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results}"
GPU_ID="${GPU_ID:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Accept model path as argument
if [ -z "$1" ]; then
    echo -e "${RED}Error: No checkpoint path provided${NC}"
    echo "Usage: $0 <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs/phi3-custom/checkpoint-1500"
    exit 1
fi

MODEL_PATH="$1"

# Validate paths
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo -e "${RED}Error: Validation file not found: $VAL_FILE${NC}"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo -e "${RED}Error: Base model not found: $BASE_MODEL${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        LLM EVALUATION PIPELINE                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Model:          $MODEL_PATH"
echo "  Base Model:     $BASE_MODEL"
echo "  Val File:       $VAL_FILE"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  GPU:            $GPU_ID"
echo "  Num Samples:    $NUM_SAMPLES"
echo ""

# Step 1: Automated Metrics
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 1/3: Automated Metrics Evaluation${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Running perplexity calculation, sample generation, and A/B comparison..."
echo ""

python evaluate_model.py \
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --gpu_id "$GPU_ID"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Automated evaluation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Automated metrics complete${NC}"
echo ""

# Step 2: LLM-as-Judge (Optional)
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 2/3: LLM-as-Judge Evaluation (Optional)${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check if API keys are set
if [ -n "$OPENAI_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "API key detected. Running LLM-as-Judge evaluation..."
    echo ""

    # Find the most recent comparison file
    COMPARISON_FILE=$(ls -t "$OUTPUT_DIR"/ab_comparison_*.json 2>/dev/null | head -1)

    if [ -n "$COMPARISON_FILE" ]; then
        JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
        MAX_COMPARISONS="${MAX_COMPARISONS:-50}"

        echo "Using judge model: $JUDGE_MODEL"
        echo "Max comparisons: $MAX_COMPARISONS"
        echo ""

        python evaluate_llm_judge.py \
            --comparison_file "$COMPARISON_FILE" \
            --judge_model "$JUDGE_MODEL" \
            --output_dir "$OUTPUT_DIR" \
            --max_comparisons "$MAX_COMPARISONS"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ LLM-as-Judge evaluation complete${NC}"
        else
            echo -e "${YELLOW}⚠ LLM-as-Judge evaluation had issues (continuing...)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No comparison file found, skipping LLM judge${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No API key found (OPENAI_API_KEY or ANTHROPIC_API_KEY)${NC}"
    echo "Skipping LLM-as-Judge evaluation."
    echo ""
    echo "To enable, set one of:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
fi

echo ""

# Step 3: Generate Summary Report
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 3/3: Generating Summary Report${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Find the latest result files
LATEST_SUMMARY=$(ls -t "$OUTPUT_DIR"/summary_*.md 2>/dev/null | head -1)
LATEST_JUDGE=$(ls -t "$OUTPUT_DIR"/judge_summary_*.md 2>/dev/null | head -1)

if [ -n "$LATEST_SUMMARY" ]; then
    echo -e "${GREEN}📊 Automated Metrics Summary:${NC}"
    echo ""
    cat "$LATEST_SUMMARY"
    echo ""
fi

if [ -n "$LATEST_JUDGE" ]; then
    echo -e "${GREEN}🤖 LLM Judge Summary:${NC}"
    echo ""
    cat "$LATEST_JUDGE"
    echo ""
fi

# Final summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        EVALUATION COMPLETE!                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo ""
echo "Files generated:"
ls -lht "$OUTPUT_DIR" | head -10
echo ""
echo -e "${GREEN}✓ Evaluation pipeline complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results in $OUTPUT_DIR"
echo "  2. Check summary files (*.md)"
echo "  3. Analyze sample generations"
echo "  4. Compare with previous evaluations"
echo ""
