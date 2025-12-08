#!/usr/bin/env python
"""
Comprehensive Dataset Sanitization Tool

Performs multiple quality checks and filters on the dataset:
1. Language detection (English-only)
2. Length validation (min/max tokens)
3. Format validation (proper message structure)
4. Content quality (repetition, empty content, etc.)
5. Encoding/character validation
6. Conversation quality metrics

Usage:
    python sanitize_dataset.py \\
        --train_file scripts/datasets/merged_train.jsonl \\
        --val_file scripts/datasets/merged_val.jsonl \\
        --output_dir scripts/datasets \\
        --filter_language \\
        --min_tokens 10 \\
        --max_tokens 1536 \\
        --max_repetition_ratio 0.5 \\
        --min_turns 2
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class DatasetSanitizer:
    def __init__(self, args):
        self.args = args
        self.stats = {
            'total': 0,
            'passed': 0,
            'filtered': defaultdict(int),
            'reasons': defaultdict(int)
        }
        
        # Load tokenizer if available
        self.tokenizer = None
        if TOKENIZER_AVAILABLE and args.model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path, 
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not LANGDETECT_AVAILABLE or not self.args.filter_language:
            return "en"  # Assume English if not filtering
        
        try:
            sample = text[:1000] if len(text) > 1000 else text
            if not sample.strip():
                return None
            return detect(sample)
        except (LangDetectException, Exception):
            return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: approximate 1 token = 4 characters
        return len(text) // 4
    
    def check_format(self, messages: List[Dict]) -> Tuple[bool, str]:
        """Validate message format"""
        if not messages:
            return False, "empty_messages"
        
        # Check for proper structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False, "invalid_message_structure"
            if "role" not in msg or "content" not in msg:
                return False, "missing_role_or_content"
            if not isinstance(msg["content"], str):
                return False, "content_not_string"
        
        # Check for proper alternation (user/assistant)
        roles = [msg["role"] for msg in messages]
        if not any(r == "user" for r in roles):
            return False, "no_user_messages"
        if not any(r == "assistant" for r in roles):
            return False, "no_assistant_messages"
        
        return True, "ok"
    
    def check_repetition(self, text: str) -> Tuple[bool, float]:
        """Check for excessive repetition"""
        if len(text) < 100:
            return True, 0.0
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) < 3:
            return True, 0.0
        
        # Count sentence frequencies
        sentence_counts = Counter(sentences)
        total_sentences = len(sentences)
        unique_sentences = len(sentence_counts)
        
        if total_sentences == 0:
            return True, 0.0
        
        repetition_ratio = 1.0 - (unique_sentences / total_sentences)
        
        # Check for repeated phrases (3+ words)
        words = text.split()
        if len(words) < 10:
            return True, repetition_ratio
        
        # Check for n-gram repetition
        ngrams = []
        for i in range(len(words) - 2):
            ngrams.append(' '.join(words[i:i+3]))
        
        if len(ngrams) > 0:
            ngram_counts = Counter(ngrams)
            max_repeat = max(ngram_counts.values()) if ngram_counts else 1
            if max_repeat > len(ngrams) * 0.1:  # More than 10% repetition
                return False, repetition_ratio
        
        max_repetition = self.args.max_repetition_ratio
        return repetition_ratio <= max_repetition, repetition_ratio
    
    def check_encoding(self, text: str) -> Tuple[bool, str]:
        """Check for encoding issues"""
        # Check for replacement characters (indicates encoding errors)
        if '\ufffd' in text or '\uFFFD' in text:
            return False, "encoding_errors"
        
        # Check for excessive control characters
        control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
        if control_chars > len(text) * 0.01:  # More than 1% control chars
            return False, "excessive_control_chars"
        
        return True, "ok"
    
    def check_content_quality(self, messages: List[Dict]) -> Tuple[bool, str]:
        """Check content quality"""
        # Combine all content
        full_text = " ".join([msg.get("content", "") for msg in messages])
        
        # Check for empty content
        if not full_text.strip():
            return False, "empty_content"
        
        # Check minimum length
        if len(full_text.strip()) < 20:
            return False, "too_short"
        
        # Check for excessive whitespace
        if len(full_text) - len(full_text.replace(' ', '')) > len(full_text) * 0.5:
            return False, "excessive_whitespace"
        
        # Check for minimum number of turns
        if len(messages) < self.args.min_turns:
            return False, "insufficient_turns"
        
        # Check for proper conversation flow (user messages should have responses)
        user_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]
        if user_indices:
            last_user_idx = user_indices[-1]
            # Should have at least one assistant message after last user message
            has_response = any(
                i > last_user_idx and msg.get("role") == "assistant"
                for i, msg in enumerate(messages)
            )
            if not has_response:
                return False, "no_response_to_user"
        
        return True, "ok"
    
    def sanitize_sequence(self, data: Dict, idx: int) -> Tuple[bool, Dict, List[str]]:
        """Run all sanitization checks on a sequence"""
        reasons = []
        messages = data.get("messages", [])
        
        # 1. Format check
        format_ok, format_reason = self.check_format(messages)
        if not format_ok:
            return False, data, [format_reason]
        
        # 2. Content quality check
        quality_ok, quality_reason = self.check_content_quality(messages)
        if not quality_ok:
            return False, data, [quality_reason]
        
        # 3. Combine text for further checks
        full_text = " ".join([msg.get("content", "") for msg in messages])
        
        # 4. Language check
        if self.args.filter_language:
            lang = self.detect_language(full_text)
            if lang != "en":
                return False, data, [f"non_english_{lang}"]
        
        # 5. Token length check
        if self.tokenizer or True:  # Always check length
            token_count = self.count_tokens(full_text)
            if token_count < self.args.min_tokens:
                return False, data, [f"too_short_tokens_{token_count}"]
            if token_count > self.args.max_tokens:
                return False, data, [f"too_long_tokens_{token_count}"]
        
        # 6. Encoding check
        encoding_ok, encoding_reason = self.check_encoding(full_text)
        if not encoding_ok:
            return False, data, [encoding_reason]
        
        # 7. Repetition check
        repetition_ok, repetition_ratio = self.check_repetition(full_text)
        if not repetition_ok:
            return False, data, [f"excessive_repetition_{repetition_ratio:.2f}"]
        
        return True, data, []
    
    def sanitize_file(self, input_file: Path, output_file: Path, dataset_name: str):
        """Sanitize a single dataset file"""
        print(f"\n{'='*60}")
        print(f"Sanitizing {dataset_name}...")
        print(f"{'='*60}")
        
        passed_sequences = []
        total = 0
        
        print("Processing sequences...")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for idx, line in enumerate(f_in):
                if idx % 10000 == 0 and idx > 0:
                    print(f"  Processed {idx:,} sequences... (passed: {len(passed_sequences):,})")
                
                try:
                    data = json.loads(line)
                    total += 1
                    self.stats['total'] += 1
                    
                    passed, sanitized_data, reasons = self.sanitize_sequence(data, idx)
                    
                    if passed:
                        passed_sequences.append(sanitized_data)
                        self.stats['passed'] += 1
                    else:
                        for reason in reasons:
                            self.stats['filtered'][reason] += 1
                            self.stats['reasons'][reason] += 1
                
                except json.JSONDecodeError:
                    self.stats['filtered']['json_decode_error'] += 1
                    self.stats['reasons']['json_decode_error'] += 1
                except Exception as e:
                    self.stats['filtered'][f'error_{type(e).__name__}'] += 1
                    self.stats['reasons'][f'error_{type(e).__name__}'] += 1
        
        # Write sanitized sequences
        print(f"\nWriting sanitized dataset to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for data in passed_sequences:
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        filtered_count = total - len(passed_sequences)
        filter_rate = (filtered_count / total * 100) if total > 0 else 0
        
        print(f"\n📊 Results for {dataset_name}:")
        print(f"  Total sequences: {total:,}")
        print(f"  Passed: {len(passed_sequences):,} ({100-filter_rate:.2f}%)")
        print(f"  Filtered: {filtered_count:,} ({filter_rate:.2f}%)")
        
        if self.stats['reasons']:
            print(f"\n  Filter reasons:")
            for reason, count in sorted(self.stats['reasons'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {reason}: {count:,}")
        
        return len(passed_sequences), filtered_count


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset sanitization")
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training JSONL file')
    parser.add_argument('--val_file', type=str, default=None,
                        help='Path to validation JSONL file (optional)')
    parser.add_argument('--output_dir', type=str, default='./datasets',
                        help='Output directory for sanitized files')
    parser.add_argument('--model_path', type=str, default='microsoft/Phi-3-medium-4k-instruct',
                        help='Model path for tokenizer (for accurate token counting)')
    
    # Filter options
    parser.add_argument('--filter_language', action='store_true',
                        help='Filter out non-English sequences')
    parser.add_argument('--min_tokens', type=int, default=10,
                        help='Minimum token count (default: 10)')
    parser.add_argument('--max_tokens', type=int, default=1536,
                        help='Maximum token count (default: 1536)')
    parser.add_argument('--min_turns', type=int, default=2,
                        help='Minimum conversation turns (default: 2)')
    parser.add_argument('--max_repetition_ratio', type=float, default=0.5,
                        help='Maximum repetition ratio (default: 0.5)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Sanitization Tool")
    print("="*60)
    print(f"Filters enabled:")
    print(f"  - Language filter: {'Yes (English only)' if args.filter_language else 'No'}")
    print(f"  - Token range: {args.min_tokens} - {args.max_tokens}")
    print(f"  - Min turns: {args.min_turns}")
    print(f"  - Max repetition: {args.max_repetition_ratio}")
    print("="*60)
    
    # Check dependencies
    if args.filter_language and not LANGDETECT_AVAILABLE:
        print("\n⚠️  Warning: langdetect not installed. Language filtering disabled.")
        print("   Install with: pip install langdetect")
        args.filter_language = False
    
    if not TOKENIZER_AVAILABLE:
        print("\n⚠️  Warning: transformers not available. Using character-based token estimation.")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize sanitizer
    sanitizer = DatasetSanitizer(args)
    
    # Sanitize training dataset
    train_input = Path(args.train_file)
    train_output = output_dir / f"{train_input.stem}_sanitized.jsonl"
    train_passed, train_filtered = sanitizer.sanitize_file(
        train_input, train_output, "TRAIN"
    )
    
    # Sanitize validation dataset if provided
    val_passed, val_filtered = 0, 0
    if args.val_file:
        val_input = Path(args.val_file)
        val_output = output_dir / f"{val_input.stem}_sanitized.jsonl"
        val_passed, val_filtered = sanitizer.sanitize_file(
            val_input, val_output, "VALIDATION"
        )
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training dataset:")
    print(f"  Passed: {train_passed:,}")
    print(f"  Filtered: {train_filtered:,}")
    if args.val_file:
        print(f"\nValidation dataset:")
        print(f"  Passed: {val_passed:,}")
        print(f"  Filtered: {val_filtered:,}")
    
    total_passed = train_passed + val_passed
    total_filtered = train_filtered + val_filtered
    total = sanitizer.stats['total']
    
    print(f"\nTotal:")
    print(f"  Passed: {total_passed:,} ({total_passed/total*100:.2f}%)")
    print(f"  Filtered: {total_filtered:,} ({total_filtered/total*100:.2f}%)")
    
    print(f"\n✓ Sanitized datasets saved to:")
    print(f"  {train_output}")
    if args.val_file:
        print(f"  {val_output}")


if __name__ == "__main__":
    main()
