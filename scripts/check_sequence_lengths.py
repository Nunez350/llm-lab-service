#!/usr/bin/env python
"""
Check dataset for sequences longer than max_seq_length

This script loads your dataset, formats it the same way as the training script,
and tokenizes it WITHOUT truncation to check how many sequences exceed your
max_seq_length threshold.

Usage:
    python check_sequence_lengths.py \\
        --model_path microsoft/Phi-3-medium-4k-instruct \\
        --train_file ./datasets/merged_train.jsonl \\
        --val_file ./datasets/merged_val.jsonl \\
        --max_seq_length 1536
"""

import argparse
import json
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from datasets import Dataset

def load_jsonl(file_path):
    """Load JSONL file with messages format"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_messages(examples, tokenizer):
    """Convert messages to text using chat template (same as training script)"""
    texts = []
    for messages in examples["messages"]:
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback formatting
            text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
                elif role == "system":
                    text += f"System: {content}\n"
        texts.append(text)
    return {"text": texts}


def check_lengths(dataset, tokenizer, max_seq_length, dataset_name):
    """Check sequence lengths in dataset"""
    print(f"\n{'='*60}")
    print(f"Checking {dataset_name}...")
    print(f"{'='*60}")
    
    # Format dataset
    print("Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_messages(x, tokenizer),
        batched=True,
        num_proc=4,
        desc=f"Formatting {dataset_name}"
    )
    
    # Tokenize WITHOUT truncation to see actual lengths
    print("Tokenizing (no truncation)...")
    def tokenize_no_truncate(examples):
        return tokenizer(
            examples["text"],
            truncation=False,  # NO truncation - we want to see actual lengths
            padding=False
        )
    
    tokenized = dataset.map(
        tokenize_no_truncate,
        batched=True,
        num_proc=4,
        desc=f"Tokenizing {dataset_name}",
        remove_columns=["messages", "text"]
    )
    
    # Get lengths
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    lengths = np.array(lengths)
    
    # Statistics
    total = len(lengths)
    over_threshold = np.sum(lengths > max_seq_length)
    over_threshold_pct = (over_threshold / total) * 100 if total > 0 else 0
    
    print(f"\n📊 Statistics for {dataset_name}:")
    print(f"  Total sequences: {total:,}")
    print(f"  Sequences > {max_seq_length} tokens: {over_threshold:,} ({over_threshold_pct:.2f}%)")
    print(f"  Sequences ≤ {max_seq_length} tokens: {total - over_threshold:,} ({(100-over_threshold_pct):.2f}%)")
    print(f"\n  Length statistics:")
    print(f"    Min: {lengths.min():,} tokens")
    print(f"    Max: {lengths.max():,} tokens")
    print(f"    Mean: {lengths.mean():.1f} tokens")
    print(f"    Median: {np.median(lengths):.1f} tokens")
    print(f"    Std: {lengths.std():.1f} tokens")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(lengths, p)
        print(f"    {p}th: {val:.1f} tokens")
    
    # Show examples of long sequences
    if over_threshold > 0:
        print(f"\n  ⚠️  Warning: {over_threshold:,} sequences exceed {max_seq_length} tokens!")
        print(f"     These will be truncated during training, potentially losing information.")
        
        # Find longest sequences
        over_indices = np.where(lengths > max_seq_length)[0]
        over_lengths = lengths[over_indices]
        sorted_indices = np.argsort(over_lengths)[::-1]  # Sort descending
        
        print(f"\n  Top 10 longest sequences (that exceed {max_seq_length}):")
        for i, idx in enumerate(sorted_indices[:10]):
            orig_idx = over_indices[idx]
            length = over_lengths[idx]
            excess = length - max_seq_length
            print(f"    {i+1}. Sequence #{orig_idx}: {length:,} tokens (exceeds by {excess:,} tokens)")
    
    # Length distribution buckets
    buckets = [
        (0, 256, "0-256"),
        (256, 512, "256-512"),
        (512, 768, "512-768"),
        (768, 1024, "768-1024"),
        (1024, 1536, "1024-1536"),
        (1536, 2048, "1536-2048"),
        (2048, 4096, "2048-4096"),
        (4096, float('inf'), "4096+")
    ]
    
    print(f"\n  Length distribution:")
    for min_len, max_len, label in buckets:
        count = np.sum((lengths >= min_len) & (lengths < max_len))
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)  # Visual bar
        print(f"    {label:12s}: {count:7,} ({pct:5.2f}%) {bar}")
    
    return {
        "total": total,
        "over_threshold": int(over_threshold),
        "over_threshold_pct": float(over_threshold_pct),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std())
    }


def main():
    parser = argparse.ArgumentParser(description="Check dataset sequence lengths")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (for tokenizer)')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation JSONL file (optional)')
    parser.add_argument('--max_seq_length', type=int, default=1536,
                        help='Maximum sequence length threshold to check against')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Sequence Length Checker")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Max sequence length threshold: {args.max_seq_length}")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and check training dataset
    train_dataset = load_jsonl(args.train_file)
    train_stats = check_lengths(train_dataset, tokenizer, args.max_seq_length, "TRAIN")
    
    # Load and check validation dataset if provided
    val_stats = None
    if args.val_file:
        val_dataset = load_jsonl(args.val_file)
        val_stats = check_lengths(val_dataset, tokenizer, args.max_seq_length, "VALIDATION")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training dataset:")
    print(f"  {train_stats['over_threshold']:,} / {train_stats['total']:,} sequences exceed {args.max_seq_length} tokens ({train_stats['over_threshold_pct']:.2f}%)")
    
    if val_stats:
        print(f"\nValidation dataset:")
        print(f"  {val_stats['over_threshold']:,} / {val_stats['total']:,} sequences exceed {args.max_seq_length} tokens ({val_stats['over_threshold_pct']:.2f}%)")
    
    if train_stats['over_threshold'] > 0 or (val_stats and val_stats['over_threshold'] > 0):
        print(f"\n⚠️  RECOMMENDATION:")
        print(f"   Consider increasing --max_seq_length if you have enough GPU memory,")
        print(f"   or pre-filter your dataset to remove very long sequences.")
    else:
        print(f"\n✓ All sequences are within {args.max_seq_length} tokens!")


if __name__ == "__main__":
    main()
