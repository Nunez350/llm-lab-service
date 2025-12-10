#!/usr/bin/env python
"""
Split dataset by sequence length - separate short and long sequences
Short sequences (<=max_length) for fast training now
Long sequences (>max_length) saved separately for later training
"""

import argparse
import json
from transformers import AutoTokenizer

def count_tokens(text, tokenizer):
    """Count tokens in text"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_dataset(input_file, output_short, output_long, max_length, model_path):
    """Split dataset into short and long sequences"""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Reading {input_file}...")
    short_sequences = []
    long_sequences = []
    total = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num:,} lines... (short: {len(short_sequences):,}, long: {len(long_sequences):,})")
            
            try:
                data = json.loads(line.strip())
                
                # Format messages into text (same as training script)
                if "messages" in data:
                    text = tokenizer.apply_chat_template(
                        data["messages"],
                        tokenize=False,
                        add_generation_prompt=False
                    )
                elif "text" in data:
                    text = data["text"]
                else:
                    continue
                
                # Count tokens
                token_count = count_tokens(text, tokenizer)
                total += 1
                
                if token_count <= max_length:
                    short_sequences.append(data)
                else:
                    long_sequences.append(data)
                    
            except json.JSONDecodeError:
                print(f"  Warning: Skipping invalid JSON at line {line_num}")
                continue
    
    # Write short sequences
    print(f"\nWriting {len(short_sequences):,} short sequences (<= {max_length} tokens) to {output_short}...")
    with open(output_short, 'w', encoding='utf-8') as f:
        for item in short_sequences:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Write long sequences
    print(f"Writing {len(long_sequences):,} long sequences (> {max_length} tokens) to {output_long}...")
    with open(output_long, 'w', encoding='utf-8') as f:
        for item in long_sequences:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Split complete!")
    print(f"  Total sequences: {total:,}")
    print(f"  Short (<= {max_length} tokens): {len(short_sequences):,} ({len(short_sequences)/total*100:.1f}%)")
    print(f"  Long (> {max_length} tokens): {len(long_sequences):,} ({len(long_sequences)/total*100:.1f}%)")
    print(f"\n  Short sequences saved to: {output_short}")
    print(f"  Long sequences saved to: {output_long}")

def main():
    parser = argparse.ArgumentParser(description="Split dataset by sequence length")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSONL file")
    parser.add_argument('--output_short', type=str, required=True, help="Output file for short sequences (<=max_length)")
    parser.add_argument('--output_long', type=str, required=True, help="Output file for long sequences (>max_length)")
    parser.add_argument('--max_length', type=int, default=1024, help="Maximum token length for 'short' sequences")
    parser.add_argument('--model_path', type=str, required=True, help="Model path for tokenizer")
    
    args = parser.parse_args()
    
    split_dataset(
        args.input_file,
        args.output_short,
        args.output_long,
        args.max_length,
        args.model_path
    )

if __name__ == "__main__":
    main()
