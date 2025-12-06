#!/usr/bin/env python3
"""
Parameterized LoRA Training Script
Can be called from Java or command line with arguments
"""
import argparse
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM with LoRA and 4-bit quantization')

    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to base model (local path or HuggingFace model ID)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='HuggingFace dataset name (e.g., "HuggingFaceH4/ultrachat_200k")')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and final model')

    # Model configuration
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length (default: 2048)')
    parser.add_argument('--load_in_4bit', type=bool, default=True,
                        help='Use 4-bit quantization (default: True)')

    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=64,
                        help='LoRA rank (default: 64)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout (default: 0.0)')
    parser.add_argument('--target_modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                        help='Comma-separated list of target modules (default: Llama/Phi-3 style)')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per-device training batch size (default: 2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Maximum training steps (overrides num_epochs if > 0)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps (default: 100)')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        choices=['linear', 'cosine', 'constant'],
                        help='Learning rate scheduler type (default: linear)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')

    # Dataset configuration
    parser.add_argument('--dataset_split', type=str, default='train_sft[:10%]',
                        help='Dataset split to use (default: train_sft[:10%])')
    parser.add_argument('--dataset_field', type=str, default='messages',
                        help='Field containing data (default: messages for conversational)')
    parser.add_argument('--num_proc', type=int, default=4,
                        help='Number of processes for dataset mapping (default: 4)')

    # Logging and saving
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Log every N steps (default: 10)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps (default: 500)')
    parser.add_argument('--save_total_limit', type=int, default=3,
                        help='Maximum number of checkpoints to keep (default: 3)')

    # GPU configuration
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID(s) to use, e.g., "0" or "0,1" (default: 0)')

    # Advanced options
    parser.add_argument('--bf16', type=bool, default=True,
                        help='Use bfloat16 precision (default: True)')
    parser.add_argument('--fp16', type=bool, default=False,
                        help='Use float16 precision (default: False)')
    parser.add_argument('--optim', type=str, default='adamw_8bit',
                        choices=['adamw_8bit', 'adamw_torch', 'adafactor'],
                        help='Optimizer (default: adamw_8bit)')

    return parser.parse_args()


def format_messages(examples, tokenizer):
    """Convert messages to text format using tokenizer's chat template."""
    texts = []
    for messages in examples["messages"]:
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback: manually format the conversation
            text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    text += f"User: {content}\n"
                else:
                    text += f"Assistant: {content}\n"
        texts.append(text)
    return {"text": texts}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print("=" * 80)
    print("LoRA Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_name} ({args.dataset_split})")
    print(f"Output: {args.output_dir}")
    print(f"GPU(s): {args.gpu_id}")
    print(f"Batch Size: {args.batch_size} × {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps} effective")
    print(f"LoRA Rank: {args.lora_r}, Alpha: {args.lora_alpha}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print("\n[1/7] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        load_in_4bit=args.load_in_4bit,
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Prepare model for k-bit training
    print("\n[2/7] Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    target_modules = args.target_modules.split(',')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\n[3/7] Loading dataset: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"Dataset size: {len(dataset)} examples")

    # Format dataset (for conversational data)
    print("\n[4/7] Formatting dataset...")
    if args.dataset_field == 'messages':
        dataset = dataset.map(
            lambda x: format_messages(x, tokenizer),
            batched=True,
            num_proc=args.num_proc,
        )

    # Tokenize dataset
    print("\n[5/7] Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    print("\n[6/7] Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else None,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=args.optim,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        dataloader_num_workers=0,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n[7/7] Starting training...")
    print("=" * 80)
    trainer.train()

    # Save
    print("\n" + "=" * 80)
    print("Training complete! Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nModel and tokenizer saved to: {args.output_dir}")
    print("=" * 80)
    print("SUCCESS!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
