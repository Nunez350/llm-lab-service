#!/usr/bin/env python
"""
Find and save sequences longer than max_seq_length

This script identifies sequences that exceed the specified max_seq_length,
saves them to a JSONL file with their original data and token counts,
so you can review and decide whether to filter them out.

Usage:
    python find_long_sequences.py \\
        --model_path microsoft/Phi-3-medium-4k-instruct \\
        --train_file ./datasets/merged_train.jsonl \\
        --val_file ./datasets/merged_val.jsonl \\
        --max_seq_length 1536 \\
        --output_file ./datasets/long_sequences.jsonl
"""

import argparse
import json
from pathlib import Path
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


def find_long_sequences(dataset, tokenizer, max_seq_length, dataset_name, output_file, dataset_type):
    """Find sequences longer than max_seq_length and save to file"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}...")
    print(f"{'='*60}")
    
    # Format dataset
    print("Formatting dataset...")
    formatted_dataset = dataset.map(
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
    
    tokenized = formatted_dataset.map(
        tokenize_no_truncate,
        batched=True,
        num_proc=4,
        desc=f"Tokenizing {dataset_name}",
        remove_columns=["messages", "text"]
    )
    
    # Get lengths and find long sequences
    print("Finding long sequences...")
    long_sequences = []
    
    for idx in range(len(dataset)):
        length = len(tokenized[idx]["input_ids"])
        if length > max_seq_length:
            # Get original data
            original_data = dataset[idx]
            
            # Get formatted text
            formatted_text = formatted_dataset[idx]["text"]
            
            # Calculate excess tokens
            excess = length - max_seq_length
            
            # Create entry with all relevant info
            entry = {
                "dataset_type": dataset_type,
                "original_index": idx,
                "token_length": length,
                "max_seq_length": max_seq_length,
                "excess_tokens": excess,
                "excess_percentage": round((excess / max_seq_length) * 100, 2),
                "original_messages": original_data.get("messages", []),
                "formatted_text_preview": formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text,
                "formatted_text_length": len(formatted_text),
                "num_messages": len(original_data.get("messages", []))
            }
            
            long_sequences.append(entry)
    
    # Sort by length (longest first)
    long_sequences.sort(key=lambda x: x["token_length"], reverse=True)
    
    # Save to file
    print(f"\nFound {len(long_sequences)} sequences longer than {max_seq_length} tokens")
    print(f"Saving to {output_file}...")
    
    # Append to file (so we can combine train and val)
    with open(output_file, 'a') as f:
        for entry in long_sequences:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(long_sequences)} sequences to {output_file}")
    
    # Print summary
    if long_sequences:
        print(f"\n  Top 10 longest sequences from {dataset_name}:")
        for i, seq in enumerate(long_sequences[:10], 1):
            print(f"    {i}. Index #{seq['original_index']}: {seq['token_length']:,} tokens "
                  f"(exceeds by {seq['excess_tokens']:,} tokens, {seq['excess_percentage']:.1f}%)")
    
    return len(long_sequences)


def main():
    parser = argparse.ArgumentParser(description="Find sequences longer than max_seq_length")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model (for tokenizer)')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation JSONL file (optional)')
    parser.add_argument('--max_seq_length', type=int, default=1536,
                        help='Maximum sequence length threshold')
    parser.add_argument('--output_file', type=str, default='./datasets/long_sequences.jsonl',
                        help='Output file to save long sequences (default: ./datasets/long_sequences.jsonl)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Find Long Sequences")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Max sequence length threshold: {args.max_seq_length}")
    print(f"Output file: {args.output_file}")
    print("="*60)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing output file if it exists
    if output_path.exists():
        print(f"\n⚠️  Removing existing output file: {args.output_file}")
        output_path.unlink()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process training dataset
    train_dataset = load_jsonl(args.train_file)
    train_count = find_long_sequences(
        train_dataset, tokenizer, args.max_seq_length, 
        "TRAIN", args.output_file, "train"
    )
    
    # Process validation dataset if provided
    val_count = 0
    if args.val_file:
        val_dataset = load_jsonl(args.val_file)
        val_count = find_long_sequences(
            val_dataset, tokenizer, args.max_seq_length,
            "VALIDATION", args.output_file, "val"
        )
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total long sequences found: {train_count + val_count}")
    print(f"  - Training: {train_count}")
    if args.val_file:
        print(f"  - Validation: {val_count}")
    print(f"\nAll sequences saved to: {args.output_file}")
    print(f"\nYou can review the file to decide which sequences to filter out.")
    print(f"Each entry contains:")
    print(f"  - original_index: Index in the original dataset")
    print(f"  - token_length: Actual token count")
    print(f"  - excess_tokens: How many tokens over the limit")
    print(f"  - original_messages: The full conversation")
    print(f"  - formatted_text_preview: Preview of formatted text")


if __name__ == "__main__":
    main()
