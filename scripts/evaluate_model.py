#!/usr/bin/env python
"""
Comprehensive LLM Evaluation Framework

Evaluates fine-tuned models using:
1. Perplexity on validation set
2. Loss metrics
3. Sample generation quality
4. Benchmark task performance
5. A/B comparison with base model

Usage:
    python evaluate_model.py \
        --model_path ./outputs/phi3-custom \
        --base_model_path /path/to/phi3-medium \
        --val_file ./datasets/merged_val.jsonl \
        --output_dir ./evaluation_results
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import Dataset
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model (LoRA adapter)')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to base model')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Path to validation JSONL file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Max tokens to generate for qualitative eval')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU to use')
    return parser.parse_args()


def load_jsonl(file_path, num_samples=None):
    """Load JSONL validation file"""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            data.append(json.loads(line))
    return data


def load_model_and_tokenizer(model_path, base_model_path, is_adapter=True):
    """Load model and tokenizer"""
    print(f"\nLoading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if is_adapter:
        print(f"Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def calculate_perplexity(model, tokenizer, examples, max_length=2048):
    """Calculate perplexity on validation set"""
    print("\n" + "="*60)
    print("CALCULATING PERPLEXITY")
    print("="*60)

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for example in tqdm(examples, desc="Calculating perplexity"):
            # Format messages
            messages = example.get("messages", [])
            if not messages:
                continue

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(model.device)

            # Calculate loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"\nAverage Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    return {
        "perplexity": float(perplexity),
        "average_loss": float(avg_loss),
        "total_tokens": total_tokens
    }


def generate_samples(model, tokenizer, examples, max_new_tokens=256, num_samples=10):
    """Generate sample responses for qualitative evaluation"""
    print("\n" + "="*60)
    print("GENERATING SAMPLE RESPONSES")
    print("="*60)

    results = []

    model.eval()
    with torch.no_grad():
        for i, example in enumerate(tqdm(examples[:num_samples], desc="Generating samples")):
            messages = example.get("messages", [])
            if not messages:
                continue

            # Extract user prompt (everything except last assistant response)
            user_messages = []
            ground_truth = None
            for msg in messages:
                if msg["role"] == "assistant" and ground_truth is None:
                    ground_truth = msg["content"]
                    break
                user_messages.append(msg)

            # Generate response
            text = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            results.append({
                "sample_id": i,
                "prompt": user_messages[-1]["content"] if user_messages else "",
                "ground_truth": ground_truth or "",
                "generated": generated_text,
                "full_conversation": messages
            })

    return results


def compare_models(base_model, finetuned_model, tokenizer, examples, num_samples=20):
    """A/B comparison between base and fine-tuned models"""
    print("\n" + "="*60)
    print("A/B MODEL COMPARISON")
    print("="*60)

    comparisons = []

    for i, example in enumerate(tqdm(examples[:num_samples], desc="Comparing models")):
        messages = example.get("messages", [])
        if not messages:
            continue

        # Extract user prompt
        user_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                break
            user_messages.append(msg)

        text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(base_model.device)

        # Generate from base model
        with torch.no_grad():
            base_outputs = base_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            base_response = tokenizer.decode(
                base_outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

        # Generate from fine-tuned model
        with torch.no_grad():
            ft_outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            ft_response = tokenizer.decode(
                ft_outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

        comparisons.append({
            "sample_id": i,
            "prompt": user_messages[-1]["content"] if user_messages else "",
            "base_response": base_response,
            "finetuned_response": ft_response
        })

    return comparisons


def save_results(output_dir, perplexity_results, sample_generations, ab_comparisons, model_info):
    """Save all evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save perplexity results
    with open(output_dir / f"perplexity_{timestamp}.json", 'w') as f:
        json.dump({
            "model_info": model_info,
            "results": perplexity_results
        }, f, indent=2)

    # Save sample generations
    with open(output_dir / f"samples_{timestamp}.json", 'w') as f:
        json.dump(sample_generations, f, indent=2)

    # Save A/B comparisons
    with open(output_dir / f"ab_comparison_{timestamp}.json", 'w') as f:
        json.dump(ab_comparisons, f, indent=2)

    # Create summary report
    summary = f"""
# Evaluation Summary
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Information
- **Fine-tuned Model:** {model_info['finetuned_path']}
- **Base Model:** {model_info['base_path']}

## Perplexity Results
- **Perplexity:** {perplexity_results['perplexity']:.4f}
- **Average Loss:** {perplexity_results['average_loss']:.4f}
- **Total Tokens Evaluated:** {perplexity_results['total_tokens']:,}

## Sample Generations
- Generated {len(sample_generations)} sample responses
- See `samples_{timestamp}.json` for details

## A/B Comparison
- Compared {len(ab_comparisons)} prompts between base and fine-tuned models
- See `ab_comparison_{timestamp}.json` for details

## Files Generated
1. `perplexity_{timestamp}.json` - Perplexity metrics
2. `samples_{timestamp}.json` - Sample generations
3. `ab_comparison_{timestamp}.json` - Base vs Fine-tuned comparison
4. `summary_{timestamp}.md` - This summary report
"""

    with open(output_dir / f"summary_{timestamp}.md", 'w') as f:
        f.write(summary)

    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - perplexity_{timestamp}.json")
    print(f"   - samples_{timestamp}.json")
    print(f"   - ab_comparison_{timestamp}.json")
    print(f"   - summary_{timestamp}.md")


def main():
    args = parse_args()

    # Set GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("="*60)
    print("LLM EVALUATION FRAMEWORK")
    print("="*60)

    # Load validation data
    print(f"\nLoading validation data from {args.val_file}...")
    val_data = load_jsonl(args.val_file, args.num_samples)
    print(f"Loaded {len(val_data)} examples")

    # Load fine-tuned model
    finetuned_model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.base_model_path,
        is_adapter=True
    )

    # Load base model for comparison
    print(f"\nLoading base model for comparison...")
    base_model, _ = load_model_and_tokenizer(
        args.base_model_path,
        args.base_model_path,
        is_adapter=False
    )

    # 1. Calculate perplexity
    perplexity_results = calculate_perplexity(
        finetuned_model,
        tokenizer,
        val_data
    )

    # 2. Generate sample responses
    sample_generations = generate_samples(
        finetuned_model,
        tokenizer,
        val_data,
        max_new_tokens=args.max_new_tokens,
        num_samples=min(20, len(val_data))
    )

    # 3. A/B comparison
    ab_comparisons = compare_models(
        base_model,
        finetuned_model,
        tokenizer,
        val_data,
        num_samples=min(20, len(val_data))
    )

    # Save all results
    model_info = {
        "finetuned_path": args.model_path,
        "base_path": args.base_model_path,
        "val_file": args.val_file,
        "num_samples": len(val_data)
    }

    save_results(
        args.output_dir,
        perplexity_results,
        sample_generations,
        ab_comparisons,
        model_info
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
