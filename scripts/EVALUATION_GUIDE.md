# 🎯 LLM Evaluation Framework Guide

Complete guide for evaluating your fine-tuned models.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Evaluation Methods](#evaluation-methods)
3. [Automated Metrics](#automated-metrics)
4. [LLM-as-Judge](#llm-as-judge)
5. [Benchmark Testing](#benchmark-testing)
6. [Human Evaluation](#human-evaluation)
7. [Interpreting Results](#interpreting-results)

---

## Quick Start

### 1. Basic Evaluation (Perplexity + Samples + A/B)

```bash
cd /home/rnu/mnt/models/llm-lab-service/scripts

python evaluate_model.py \
  --model_path ./outputs/phi3-custom \
  --base_model_path /home/rnu/mnt/models/phi_models/phi3-medium \
  --val_file ./datasets/merged_val.jsonl \
  --output_dir ./evaluation_results \
  --num_samples 100 \
  --gpu_id 0
```

**What it does:**
- ✅ Calculates perplexity on validation set
- ✅ Generates sample responses for review
- ✅ Creates A/B comparison (base vs fine-tuned)
- ⏱️ **Time:** ~10-15 minutes for 100 samples

**Output files:**
```
evaluation_results/
├── perplexity_20241206_123456.json
├── samples_20241206_123456.json
├── ab_comparison_20241206_123456.json
└── summary_20241206_123456.md
```

---

### 2. LLM-as-Judge Evaluation (Advanced)

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"

# Run LLM judge
python evaluate_llm_judge.py \
  --comparison_file ./evaluation_results/ab_comparison_*.json \
  --judge_model gpt-4o \
  --output_dir ./evaluation_results \
  --max_comparisons 50
```

**Judge Models:**
- `gpt-4o` - Fast, cost-effective (~$0.50 for 50 comparisons)
- `gpt-4` - More thorough (~$2.00 for 50 comparisons)
- `claude-3-sonnet` - Good balance (~$1.00 for 50 comparisons)
- `claude-3-opus` - Most thorough (~$4.00 for 50 comparisons)

**Output files:**
```
evaluation_results/
├── llm_judge_20241206_123456.json
└── judge_summary_20241206_123456.md
```

---

## Evaluation Methods

### Method 1: Automated Metrics ⚡

**Best for:** Quick, objective evaluation

**Metrics:**
- **Perplexity**: Lower is better (measures prediction confidence)
  - Good: < 20
  - Acceptable: 20-50
  - Poor: > 50

- **Loss**: Lower is better
  - Good: < 1.5
  - Acceptable: 1.5-2.5
  - Poor: > 2.5

**Pros:**
- ✅ Fast and cheap
- ✅ Reproducible
- ✅ Easy to compare across models

**Cons:**
- ❌ Doesn't measure helpfulness
- ❌ May not reflect real-world performance

---

### Method 2: LLM-as-Judge 🤖

**Best for:** Comprehensive quality assessment

**Evaluation Criteria:**
1. **Helpfulness** (1-10): Does it help the user?
2. **Accuracy** (1-10): Is the information correct?
3. **Coherence** (1-10): Is it well-structured?
4. **Relevance** (1-10): Does it stay on topic?
5. **Overall Quality** (1-10): General assessment

**Pros:**
- ✅ Evaluates real quality
- ✅ Catches nuanced issues
- ✅ Scales well

**Cons:**
- ❌ Requires API costs
- ❌ Can have bias
- ❌ Not 100% reliable

---

### Method 3: Benchmark Testing 📊

**Standard Benchmarks:**

1. **MMLU** (Massive Multitask Language Understanding)
   - 57 subjects
   - Multiple choice
   - Good: > 60%

2. **HellaSwag** (Common Sense Reasoning)
   - Good: > 75%

3. **TruthfulQA** (Factuality)
   - Good: > 40%

4. **GSM8K** (Math Problems)
   - Good: > 50%

**Run benchmarks:**
```bash
# Using lm-evaluation-harness
pip install lm-eval

lm_eval --model hf \
  --model_args pretrained=./outputs/phi3-custom \
  --tasks mmlu,hellaswag,truthfulqa_mc,gsm8k \
  --batch_size 8 \
  --output_path ./benchmark_results
```

---

### Method 4: Human Evaluation 👥

**Best for:** Final validation before deployment

**Process:**
1. Generate 50-100 diverse test cases
2. Have 2-3 humans evaluate each response
3. Use scoring rubric (1-5 scale)
4. Calculate inter-rater agreement

**Template for evaluators:**
```
Prompt: [User question]

Response A (Base):
[Base model response]

Response B (Fine-tuned):
[Fine-tuned response]

Rate each (1-5):
□ Helpfulness: ___
□ Accuracy: ___
□ Clarity: ___
□ Overall: ___

Which is better? [ ] A  [ ] B  [ ] Tie
```

---

## Interpreting Results

### Perplexity Scores

| Perplexity | Interpretation |
|------------|----------------|
| < 10 | Excellent - Very confident predictions |
| 10-20 | Good - Model is well-calibrated |
| 20-50 | Acceptable - Some uncertainty |
| > 50 | Poor - High uncertainty, needs more training |

### LLM Judge Win Rates

| Win Rate | Interpretation |
|----------|----------------|
| > 70% | Excellent - Clear improvement |
| 50-70% | Good - Noticeable improvement |
| 40-50% | Marginal - Slight improvement |
| < 40% | Poor - Fine-tuning may have hurt performance |

### Score Improvements

| Δ Score | Interpretation |
|---------|----------------|
| > +2.0 | Excellent improvement |
| +1.0 to +2.0 | Good improvement |
| +0.5 to +1.0 | Marginal improvement |
| < +0.5 | Negligible change |
| Negative | Regression (investigate!) |

---

## Evaluation Workflow

### Full Evaluation Pipeline

```bash
#!/bin/bash
# complete_evaluation.sh

MODEL_PATH="./outputs/phi3-custom"
BASE_MODEL="/home/rnu/mnt/models/phi_models/phi3-medium"
VAL_FILE="./datasets/merged_val.jsonl"
OUTPUT_DIR="./evaluation_results"

echo "🔍 Step 1: Running automated metrics..."
python evaluate_model.py \
  --model_path $MODEL_PATH \
  --base_model_path $BASE_MODEL \
  --val_file $VAL_FILE \
  --output_dir $OUTPUT_DIR \
  --num_samples 100

echo "🤖 Step 2: LLM-as-Judge evaluation..."
COMPARISON_FILE=$(ls -t $OUTPUT_DIR/ab_comparison_*.json | head -1)
python evaluate_llm_judge.py \
  --comparison_file $COMPARISON_FILE \
  --judge_model gpt-4o \
  --output_dir $OUTPUT_DIR \
  --max_comparisons 50

echo "✅ Evaluation complete! Check $OUTPUT_DIR for results."
```

**Make it executable:**
```bash
chmod +x complete_evaluation.sh
./complete_evaluation.sh
```

---

## Cost Estimates

### API Costs (LLM-as-Judge)

| Judge Model | Cost per 100 comparisons | Speed |
|-------------|---------------------------|-------|
| GPT-4o | ~$1.00 | Fast |
| GPT-4 | ~$4.00 | Medium |
| Claude Sonnet | ~$2.00 | Medium |
| Claude Opus | ~$8.00 | Slow |

### GPU Costs (Automated)

| Task | GPU Hours | Cost (H100) |
|------|-----------|-------------|
| Perplexity (1K samples) | 0.1h | ~$0.20 |
| Sample generation (100) | 0.2h | ~$0.40 |
| A/B comparison (100) | 0.3h | ~$0.60 |

---

## Best Practices

### 1. Start Small
```bash
# Test with 10 samples first
python evaluate_model.py --num_samples 10
```

### 2. Use Diverse Test Sets
- Include different question types
- Mix easy and hard questions
- Cover multiple domains

### 3. Track Over Time
```bash
# Create experiment tracking
mkdir -p experiments/exp_001_baseline
mv evaluation_results/* experiments/exp_001_baseline/

# Next experiment
mkdir -p experiments/exp_002_more_data
```

### 4. Document Everything
```markdown
# experiments/exp_001_baseline/README.md

## Configuration
- Dataset: OpenHermes + SlimOrca (450K samples)
- LoRA rank: 64
- Learning rate: 2e-4
- Epochs: 3

## Results
- Perplexity: 18.4
- Win rate vs base: 68%
- Overall score improvement: +1.8

## Notes
- Good performance on factual questions
- Struggles with creative writing
```

---

## Troubleshooting

### Issue: High perplexity (>50)
**Solutions:**
- Train for more epochs
- Increase dataset size
- Check data quality
- Reduce learning rate

### Issue: LLM judge shows regression
**Solutions:**
- Check for overfitting (compare train vs val loss)
- Review sample outputs manually
- Consider curriculum learning
- Adjust LoRA hyperparameters

### Issue: Inconsistent results
**Solutions:**
- Increase evaluation sample size
- Use multiple judge models
- Set random seeds
- Average over multiple runs

---

## Example Results

### Good Fine-tuning
```
Perplexity: 15.2 (Base: 22.1)
Win Rate: 72%
Avg Score: 7.8/10 (Base: 6.2/10)
Improvement: +1.6 points
```
✅ **Clear improvement across all metrics**

### Marginal Fine-tuning
```
Perplexity: 19.8 (Base: 20.5)
Win Rate: 52%
Avg Score: 6.5/10 (Base: 6.2/10)
Improvement: +0.3 points
```
⚠️ **Small improvement - consider more training**

### Failed Fine-tuning
```
Perplexity: 28.4 (Base: 20.1)
Win Rate: 35%
Avg Score: 5.8/10 (Base: 6.3/10)
Improvement: -0.5 points
```
❌ **Regression - investigate immediately**

---

## Next Steps After Evaluation

### If Results Are Good (Win rate > 60%)
1. ✅ Run full benchmark suite
2. ✅ Human evaluation (50-100 samples)
3. ✅ Deploy to staging environment
4. ✅ A/B test with real users

### If Results Are Marginal (Win rate 40-60%)
1. 🔄 Train for more epochs
2. 🔄 Increase dataset size
3. 🔄 Try different hyperparameters
4. 🔄 Add more diverse data

### If Results Are Poor (Win rate < 40%)
1. 🔴 Check data quality
2. 🔴 Review training logs
3. 🔴 Verify data preprocessing
4. 🔴 Consider different base model

---

## Questions?

Check the main README or create an issue on GitHub:
https://github.com/Nunez350/llm-lab-service
