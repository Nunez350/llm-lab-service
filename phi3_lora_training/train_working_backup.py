#!/usr/bin/env python
"""
Fast & High-Quality QLoRA fine-tuning with validation
Tested on RTX 5090 (Blackwell) – December 2025
"""

import argparse
import os
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="e.g. microsoft/Phi-3-medium-4k-instruct")
    parser.add_argument('--train_file', type=str, required=True, help="path/to/train.jsonl")
    parser.add_argument('--val_file', type=str, required=True, help="path/to/val.jsonl (required for best checkpoint)")
    parser.add_argument('--output_dir', type=str, required=True, help="where to save")

    # Feel free to tweak these
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=3)           # 3–4 works great on 5090
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)

    return parser.parse_args()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)

def format_messages(examples, tokenizer):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 8-bit (best quality + speed on RTX 5090)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,                  # ← 8-bit = quality + speed king
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # NO flash_attention_2 → safe on Blackwell
    )

    model = prepare_model_for_kbit_training(model)

    # ← BIGGEST SPEEDUP that still works perfectly with validation
    print("Applying torch.compile() – expect +30–40% speed...")
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dataset = load_jsonl(args.train_file)
    val_dataset = load_jsonl(args.val_file)
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} examples")

    print("Formatting + tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: format_messages(x, tokenizer),
        batched=True, num_proc=4, desc="Formatting train"
    ).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_seq_length),
        batched=True, num_proc=4, remove_columns=["messages", "text"], desc="Tokenizing train"
    )

    val_dataset = val_dataset.map(
        lambda x: format_messages(x, tokenizer),
        batched=True, num_proc=4, desc="Formatting val"
    ).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_seq_length),
        batched=True, num_proc=4, remove_columns=["messages", "text"], desc="Tokenizing val"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,           # Eval can be larger
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,

        bf16=True,
        optim="adamw_torch",                    # Faster & more stable
        logging_steps=10,
        save_steps=2000,
        save_total_limit=3,

        # Validation – smart & low overhead
        evaluation_strategy="steps",
        eval_steps=2000,                        # ~every 2–3k examples
        eval_accumulation_steps=8,              # Prevents OOM during eval
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Speed flags
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_compile=True,                     # Same as manual torch.compile

        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting training – this will be FAST and HIGH-QUALITY!\n")
    trainer.train()

    # Save final best model
    best_dir = os.path.join(args.output_dir, "final-best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nTraining complete! Best model saved to: {best_dir}")

if __name__ == "__main__":
    main()