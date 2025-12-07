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
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--gpu_id', type=str, default='0')

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

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Disable cache to avoid DynamicCache compatibility issues during training and eval
    model.config.use_cache = False

    # Apply torch.compile for 10-20% speedup (requires PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("✓ Torch compile enabled")
    except Exception as e:
        print(f"⚠ Torch compile not available: {e}")

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

    model = get_peft_model(model, lora_config)
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

    # Tokenize
    print("Tokenizing...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False
        )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Parallel processing for 3-4x speedup
        remove_columns=["messages", "text"],
        desc="Tokenizing train"
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,  # Parallel processing for 3-4x speedup
            remove_columns=["messages", "text"],
            desc="Tokenizing val"
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
        dataloader_num_workers=0,  # Avoid worker process overhead (30-40% faster)
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=1000 if val_dataset else None,  # Reduced from 500 (saves ~1.75 hours)
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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
