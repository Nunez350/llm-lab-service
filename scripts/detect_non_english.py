#!/usr/bin/env python
"""
Detect and filter non-English sequences from dataset

Uses langdetect library to identify non-English sequences.
Can optionally filter them out and create new English-only datasets.

Usage:
    python detect_non_english.py \\
        --train_file ./datasets/merged_train.jsonl \\
        --val_file ./datasets/merged_val.jsonl \\
        --output_dir ./datasets \\
        [--filter]  # Actually filter them out and create new files
"""

import argparse
import json
from pathlib import Path
from collections import Counter

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Install with: pip install langdetect")

def detect_language(text):
    """Detect language of text"""
    if not LANGDETECT_AVAILABLE:
        return None
    
    try:
        # Use first 1000 chars for faster detection
        sample = text[:1000] if len(text) > 1000 else text
        if not sample.strip():
            return None
        return detect(sample)
    except LangDetectException:
        return None
    except Exception:
        return None

def analyze_dataset(file_path, dataset_name, filter_out=False, output_dir=None):
    """Analyze dataset for non-English sequences"""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}...")
    print(f"{'='*60}")
    
    if not LANGDETECT_AVAILABLE:
        print("ERROR: langdetect not available. Install with: pip install langdetect")
        return None, None
    
    languages = []
    non_english_indices = []
    english_indices = []
    total_chars = 0
    non_english_chars = 0
    
    print("Processing sequences...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0 and idx > 0:
                print(f"  Processed {idx:,} sequences...")
            
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # Combine all message content
            full_text = " ".join([msg.get("content", "") for msg in messages])
            total_chars += len(full_text)
            
            # Detect language
            lang = detect_language(full_text)
            languages.append(lang)
            
            if lang != "en":
                non_english_indices.append(idx)
                non_english_chars += len(full_text)
            else:
                english_indices.append(idx)
    
    # Statistics
    total = len(languages)
    lang_counts = Counter(languages)
    non_english_count = len(non_english_indices)
    english_count = len(english_indices)
    
    print(f"\n📊 Language Statistics for {dataset_name}:")
    print(f"  Total sequences: {total:,}")
    print(f"  English (en): {english_count:,} ({english_count/total*100:.2f}%)")
    print(f"  Non-English: {non_english_count:,} ({non_english_count/total*100:.2f}%)")
    print(f"\n  Character distribution:")
    print(f"    Total characters: {total_chars:,}")
    print(f"    Non-English chars: {non_english_chars:,} ({non_english_chars/total_chars*100:.2f}%)")
    
    print(f"\n  Top 10 languages:")
    for lang, count in lang_counts.most_common(10):
        pct = count / total * 100
        print(f"    {lang or 'unknown':8s}: {count:7,} ({pct:5.2f}%)")
    
    # Show examples of non-English sequences
    if non_english_count > 0:
        print(f"\n  Examples of non-English sequences (first 5):")
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx in non_english_indices[:5]:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    first_msg = messages[0].get("content", "")[:100] if messages else ""
                    lang = languages[idx]
                    print(f"    Index {idx} ({lang}): {first_msg}...")
    
    # Filter if requested
    if filter_out and output_dir:
        output_path = Path(output_dir) / f"{Path(file_path).stem}_english_only.jsonl"
        print(f"\n  Filtering out non-English sequences...")
        print(f"  Writing English-only dataset to: {output_path}")
        
        english_count = 0
        with open(file_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            for idx, line in enumerate(f_in):
                if idx in english_indices:
                    f_out.write(line)
                    english_count += 1
        
        print(f"  ✓ Saved {english_count:,} English-only sequences to {output_path}")
        return output_path, non_english_count
    
    return None, non_english_count

def main():
    parser = argparse.ArgumentParser(description="Detect non-English sequences in dataset")
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation JSONL file (optional)')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                        help='Output directory for filtered files (if --filter is used)')
    parser.add_argument('--filter', action='store_true',
                        help='Actually filter out non-English sequences and create new files')
    
    args = parser.parse_args()
    
    if not LANGDETECT_AVAILABLE:
        print("ERROR: langdetect library is required.")
        print("Install it with: pip install langdetect")
        return
    
    print("="*60)
    print("Non-English Sequence Detector")
    print("="*60)
    
    if args.filter:
        print("Mode: FILTERING (will create new English-only files)")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        print("Mode: ANALYSIS ONLY (use --filter to actually filter)")
    
    # Analyze training dataset
    train_output, train_non_english = analyze_dataset(
        args.train_file, "TRAIN", args.filter, args.output_dir
    )
    
    # Analyze validation dataset if provided
    val_output = None
    val_non_english = 0
    if args.val_file:
        val_output, val_non_english = analyze_dataset(
            args.val_file, "VALIDATION", args.filter, args.output_dir
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training dataset:")
    print(f"  Non-English sequences: {train_non_english:,}")
    if args.filter and train_output:
        print(f"  English-only file: {train_output}")
    
    if args.val_file:
        print(f"\nValidation dataset:")
        print(f"  Non-English sequences: {val_non_english:,}")
        if args.filter and val_output:
            print(f"  English-only file: {val_output}")
    
    total_non_english = train_non_english + val_non_english
    print(f"\nTotal non-English sequences: {total_non_english:,}")
    
    if args.filter:
        print(f"\n✓ Filtering complete! New English-only datasets created.")
    else:
        print(f"\n💡 Tip: Use --filter to create English-only versions of your datasets")

if __name__ == "__main__":
    main()
