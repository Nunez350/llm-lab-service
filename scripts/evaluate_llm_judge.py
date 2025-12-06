#!/usr/bin/env python
"""
LLM-as-Judge Evaluation

Uses a powerful LLM (GPT-4, Claude, or local model) to evaluate
the quality of fine-tuned model responses.

Evaluates on:
1. Helpfulness
2. Accuracy
3. Coherence
4. Relevance
5. Overall quality

Usage:
    python evaluate_llm_judge.py \
        --comparison_file ./evaluation_results/ab_comparison_*.json \
        --judge_model gpt-4 \
        --output_dir ./evaluation_results
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import openai  # pip install openai
import anthropic  # pip install anthropic


JUDGE_PROMPT_TEMPLATE = """You are an expert AI evaluator. Compare two AI assistant responses to the same user prompt.

User Prompt:
{prompt}

Response A (Base Model):
{response_a}

Response B (Fine-tuned Model):
{response_b}

Evaluate both responses on the following criteria (score 1-10):
1. **Helpfulness**: Does it address the user's question/request?
2. **Accuracy**: Is the information correct and factual?
3. **Coherence**: Is it well-structured and easy to follow?
4. **Relevance**: Does it stay on topic?
5. **Overall Quality**: Overall assessment

For each criterion, provide:
- Score for Response A (1-10)
- Score for Response B (1-10)
- Brief explanation

Then provide a final verdict: Which response is better overall (A, B, or Tie)?

Respond in JSON format:
{{
  "helpfulness": {{"a": score, "b": score, "explanation": "..."}},
  "accuracy": {{"a": score, "b": score, "explanation": "..."}},
  "coherence": {{"a": score, "b": score, "explanation": "..."}},
  "relevance": {{"a": score, "b": score, "explanation": "..."}},
  "overall": {{"a": score, "b": score, "explanation": "..."}},
  "verdict": "A" | "B" | "Tie",
  "reasoning": "Overall reasoning for the verdict"
}}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison_file', type=str, required=True,
                        help='Path to A/B comparison JSON file')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                        choices=['gpt-4', 'gpt-4o', 'claude-3-opus', 'claude-3-sonnet'],
                        help='Judge model to use')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--max_comparisons', type=int, default=None,
                        help='Maximum number of comparisons to evaluate')
    return parser.parse_args()


def call_openai_judge(prompt, model="gpt-4o", api_key=None):
    """Call OpenAI API for judging"""
    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert AI evaluator. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def call_anthropic_judge(prompt, model="claude-3-sonnet-20240229", api_key=None):
    """Call Anthropic API for judging"""
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.3,
        system="You are an expert AI evaluator. Always respond with valid JSON.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract JSON from response
    content = response.content[0].text
    # Find JSON block
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def evaluate_with_judge(comparisons, judge_model, api_key=None, max_comparisons=None):
    """Evaluate all comparisons with LLM judge"""
    print("\n" + "="*60)
    print(f"EVALUATING WITH {judge_model.upper()}")
    print("="*60)

    results = []

    if max_comparisons:
        comparisons = comparisons[:max_comparisons]

    for i, comparison in enumerate(tqdm(comparisons, desc="Judging responses")):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=comparison['prompt'],
            response_a=comparison['base_response'],
            response_b=comparison['finetuned_response']
        )

        try:
            if judge_model.startswith('gpt'):
                judgment = call_openai_judge(prompt, model=judge_model, api_key=api_key)
            elif judge_model.startswith('claude'):
                judgment = call_anthropic_judge(prompt, model=judge_model, api_key=api_key)
            else:
                raise ValueError(f"Unsupported judge model: {judge_model}")

            results.append({
                "sample_id": comparison['sample_id'],
                "prompt": comparison['prompt'],
                "judgment": judgment
            })

        except Exception as e:
            print(f"\n⚠️  Error evaluating sample {i}: {e}")
            results.append({
                "sample_id": comparison['sample_id'],
                "prompt": comparison['prompt'],
                "judgment": {"error": str(e)}
            })

    return results


def calculate_statistics(judgments):
    """Calculate win rates and average scores"""
    stats = {
        "total_comparisons": len(judgments),
        "base_wins": 0,
        "finetuned_wins": 0,
        "ties": 0,
        "avg_scores": {
            "base": {"helpfulness": 0, "accuracy": 0, "coherence": 0, "relevance": 0, "overall": 0},
            "finetuned": {"helpfulness": 0, "accuracy": 0, "coherence": 0, "relevance": 0, "overall": 0}
        }
    }

    valid_judgments = [j for j in judgments if "error" not in j["judgment"]]

    for judgment in valid_judgments:
        j = judgment["judgment"]

        # Count verdicts
        verdict = j.get("verdict", "Tie")
        if verdict == "A":
            stats["base_wins"] += 1
        elif verdict == "B":
            stats["finetuned_wins"] += 1
        else:
            stats["ties"] += 1

        # Accumulate scores
        for criterion in ["helpfulness", "accuracy", "coherence", "relevance", "overall"]:
            if criterion in j:
                stats["avg_scores"]["base"][criterion] += j[criterion].get("a", 0)
                stats["avg_scores"]["finetuned"][criterion] += j[criterion].get("b", 0)

    # Calculate averages
    n = len(valid_judgments) or 1
    for criterion in stats["avg_scores"]["base"]:
        stats["avg_scores"]["base"][criterion] /= n
        stats["avg_scores"]["finetuned"][criterion] /= n

    # Calculate win rates
    stats["base_win_rate"] = stats["base_wins"] / n * 100
    stats["finetuned_win_rate"] = stats["finetuned_wins"] / n * 100
    stats["tie_rate"] = stats["ties"] / n * 100

    return stats


def save_judge_results(output_dir, judgments, stats, model_info):
    """Save LLM judge results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed judgments
    with open(output_dir / f"llm_judge_{timestamp}.json", 'w') as f:
        json.dump({
            "model_info": model_info,
            "statistics": stats,
            "judgments": judgments
        }, f, indent=2)

    # Create summary report
    summary = f"""
# LLM-as-Judge Evaluation Report
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Judge Model:** {model_info['judge_model']}

## Win Rates
- **Base Model Wins:** {stats['base_wins']} ({stats['base_win_rate']:.1f}%)
- **Fine-tuned Model Wins:** {stats['finetuned_wins']} ({stats['finetuned_win_rate']:.1f}%)
- **Ties:** {stats['ties']} ({stats['tie_rate']:.1f}%)

## Average Scores (1-10 scale)

### Base Model
- Helpfulness: {stats['avg_scores']['base']['helpfulness']:.2f}
- Accuracy: {stats['avg_scores']['base']['accuracy']:.2f}
- Coherence: {stats['avg_scores']['base']['coherence']:.2f}
- Relevance: {stats['avg_scores']['base']['relevance']:.2f}
- Overall: {stats['avg_scores']['base']['overall']:.2f}

### Fine-tuned Model
- Helpfulness: {stats['avg_scores']['finetuned']['helpfulness']:.2f}
- Accuracy: {stats['avg_scores']['finetuned']['accuracy']:.2f}
- Coherence: {stats['avg_scores']['finetuned']['coherence']:.2f}
- Relevance: {stats['avg_scores']['finetuned']['relevance']:.2f}
- Overall: {stats['avg_scores']['finetuned']['overall']:.2f}

## Improvement Metrics
- Helpfulness: {stats['avg_scores']['finetuned']['helpfulness'] - stats['avg_scores']['base']['helpfulness']:+.2f}
- Accuracy: {stats['avg_scores']['finetuned']['accuracy'] - stats['avg_scores']['base']['accuracy']:+.2f}
- Coherence: {stats['avg_scores']['finetuned']['coherence'] - stats['avg_scores']['base']['coherence']:+.2f}
- Relevance: {stats['avg_scores']['finetuned']['relevance'] - stats['avg_scores']['base']['relevance']:+.2f}
- Overall: {stats['avg_scores']['finetuned']['overall'] - stats['avg_scores']['base']['overall']:+.2f}

## Files Generated
- `llm_judge_{timestamp}.json` - Detailed judgments
- `judge_summary_{timestamp}.md` - This summary
"""

    with open(output_dir / f"judge_summary_{timestamp}.md", 'w') as f:
        f.write(summary)

    print(f"\n✅ Judge results saved to: {output_dir}")
    print(f"   - llm_judge_{timestamp}.json")
    print(f"   - judge_summary_{timestamp}.md")

    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n🏆 Win Rates:")
    print(f"   Base Model: {stats['base_wins']} wins ({stats['base_win_rate']:.1f}%)")
    print(f"   Fine-tuned: {stats['finetuned_wins']} wins ({stats['finetuned_win_rate']:.1f}%)")
    print(f"   Ties: {stats['ties']} ({stats['tie_rate']:.1f}%)")

    print(f"\n📊 Overall Score Improvement:")
    improvement = stats['avg_scores']['finetuned']['overall'] - stats['avg_scores']['base']['overall']
    print(f"   {improvement:+.2f} points (Base: {stats['avg_scores']['base']['overall']:.2f} → Fine-tuned: {stats['avg_scores']['finetuned']['overall']:.2f})")


def main():
    args = parse_args()

    print("="*60)
    print("LLM-AS-JUDGE EVALUATION")
    print("="*60)

    # Load comparison data
    print(f"\nLoading comparisons from {args.comparison_file}...")
    with open(args.comparison_file, 'r') as f:
        comparisons = json.load(f)

    print(f"Loaded {len(comparisons)} comparisons")

    # Evaluate with LLM judge
    judgments = evaluate_with_judge(
        comparisons,
        args.judge_model,
        api_key=args.api_key,
        max_comparisons=args.max_comparisons
    )

    # Calculate statistics
    stats = calculate_statistics(judgments)

    # Save results
    model_info = {
        "judge_model": args.judge_model,
        "comparison_file": args.comparison_file,
        "num_comparisons": len(comparisons)
    }

    save_judge_results(args.output_dir, judgments, stats, model_info)

    print("\n" + "="*60)
    print("LLM JUDGE EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
