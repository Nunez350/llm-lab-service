#!/usr/bin/env python
"""
Train with custom prepared dataset (JSONL format)

Usage:
    python train_custom_dataset.py \\
        --model_path /path/to/model \\
        --train_file ./datasets/merged_train.jsonl \\
        --val_file ./datasets/merged_val.jsonl \\
        --output_dir ./phi3-custom
"""

import argparse
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Optional
    parser.add_argument('--val_file', type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=3)  # Optimal: 3 works, 4 causes OOM
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--gpu_id', type=str, default='0')
    
    # Speed optimization options
    parser.add_argument('--dataloader_num_workers', type=int, default=2,
                        help='Number of dataloader workers (default: 2, 0=disabled)')
    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument('--dataloader_pin_memory', type=str_to_bool, default=True,
                        help='Pin memory for faster GPU transfer (default: True)')
    parser.add_argument('--optimizer', type=str, default='adamw_8bit',
                        choices=['adamw_8bit', 'adamw_torch', 'adamw_torch_fused', 'adafactor'],
                        help='Optimizer type (default: adamw_8bit, faster for 8-bit models)')
    parser.add_argument('--cache_dataset', action='store_true',
                        help='Cache tokenized dataset to disk to avoid re-tokenization')
    parser.add_argument('--eval_steps', type=int, default=5000,
                        help='Number of steps between evaluations (default: 5000)')

    return parser.parse_args()


def load_jsonl(file_path):
    """Load JSONL file with messages format"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_messages(examples, tokenizer):
    """Convert messages to text using chat template"""
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


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("="*60)
    print("Loading model and tokenizer...")
    print("="*60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 8-bit quantization (faster on RTX 5090, reduces memory usage)
    # Using load_in_8bit=True for compatibility with current PEFT/bitsandbytes versions
    # Note: BitsAndBytesConfig causes compatibility issues with PEFT 0.11.1
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Disable cache to avoid DynamicCache compatibility issues during training and eval
    model.config.use_cache = False

    # Workaround for PEFT 0.11.1 compatibility with bitsandbytes
    # Fix missing memory_efficient_backward attribute in MatmulLtState
    try:
        import bitsandbytes as bnb
        # Patch all Linear8bitLt modules to have the required state attributes
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear8bitLt):
                if hasattr(module, 'state') and module.state is not None:
                    # Add missing attribute to MatmulLtState if it doesn't exist
                    if not hasattr(module.state, 'memory_efficient_backward'):
                        setattr(module.state, 'memory_efficient_backward', False)
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Warning: Could not patch bitsandbytes compatibility: {e}")
        print("This may cause issues with PEFT. Consider updating bitsandbytes or PEFT versions.")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA with error handling for compatibility issues
    try:
        model = get_peft_model(model, lora_config)
    except AttributeError as e:
        if 'memory_efficient_backward' in str(e):
            print("\n" + "="*60)
            print("ERROR: PEFT/bitsandbytes compatibility issue detected")
            print("="*60)
            print("The error suggests a version mismatch between PEFT and bitsandbytes.")
            print("\nTroubleshooting steps:")
            print("1. Check your bitsandbytes version: pip show bitsandbytes")
            print("2. Try downgrading bitsandbytes: pip install bitsandbytes==0.41.3")
            print("3. Or try upgrading PEFT: pip install peft==0.12.0")
            print("4. If using PEFT 0.12.0+, you may need to upgrade transformers to 4.45.0+")
            print("\nCurrent versions in requirements.txt:")
            print("  - peft==0.11.1")
            print("  - bitsandbytes>=0.40.0")
            print("="*60)
        raise
    model.print_trainable_parameters()

    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)

    # Load training data
    train_dataset = load_jsonl(args.train_file)
    print(f"Training examples: {len(train_dataset):,}")

    # Load validation data if provided
    val_dataset = None
    if args.val_file:
        val_dataset = load_jsonl(args.val_file)
        print(f"Validation examples: {len(val_dataset):,}")

    # Format datasets
    print("\nFormatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_messages(x, tokenizer),
        batched=True,
        num_proc=4,  # Parallel processing for 3-4x speedup
        desc="Formatting train"
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: format_messages(x, tokenizer),
            batched=True,
            num_proc=4,  # Parallel processing for 3-4x speedup
            desc="Formatting val"
        )

    # Tokenize (with optional caching)
    print("Tokenizing...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False
        )

    # Cache tokenized datasets if requested (saves time on subsequent runs)
    cache_dir = None
    if args.cache_dataset:
        cache_dir = os.path.join(args.output_dir, ".dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✓ Dataset caching enabled: {cache_dir}")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Parallel processing for 3-4x speedup
        remove_columns=["messages", "text"],
        desc="Tokenizing train",
        cache_file_name=os.path.join(cache_dir, "train_tokenized.arrow") if cache_dir else None
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,  # Parallel processing for 3-4x speedup
            remove_columns=["messages", "text"],
            desc="Tokenizing val",
            cache_file_name=os.path.join(cache_dir, "val_tokenized.arrow") if cache_dir else None
        )

    # Training arguments (optimized for RTX 5090)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        bf16=True,
        optim=args.optimizer,  # Use optimized optimizer (adamw_8bit is faster for quantized models)
        logging_steps=10,
        save_steps=args.eval_steps,  # Must be a multiple of eval_steps for load_best_model_at_end
        save_total_limit=3,

        # Dataloader optimizations (enable for faster data loading)
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_prefetch_factor=2,  # Prefetch batches for faster loading

        # Validation settings
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        eval_accumulation_steps=8 if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,

        # torch_compile is not compatible with quantized models + PEFT
        # torch_compile=False,
        report_to="none",
    )

    # Data collator (optimized with padding to multiple of 8 for better GPU utilization)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for GPU tensor operations
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    trainer.train()

    print("\n" + "="*60)
    print("Saving model...")
    print("="*60)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nModel saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
