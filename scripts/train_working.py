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
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="e.g. microsoft/Phi-3-medium-4k-instruct")
    parser.add_argument('--train_file', type=str, required=True, help="path/to/train.jsonl")
    parser.add_argument('--val_file', type=str, default=None, help="path/to/val.jsonl (optional, enables validation and best checkpoint selection)")
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

    print("Loading model in 4-bit (QLoRA for speed + memory efficiency)...")
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # For multi-GPU DDP with quantized models, we need special handling
    # Check if we're in a distributed environment
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        # Multi-GPU DDP: set CUDA device first, then load without device_map
        # DDP will handle device placement
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device_map = None  # Let DDP handle device placement
    else:
        # Single GPU: use explicit device to avoid spreading across GPUs
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",    # Phi-3 requirement
    )

    # Prepare model for k-bit training - this enables gradients for quantized models
    # and is required for gradient checkpointing with 4-bit quantization
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config - 2025 canonical list for Phi-3 (includes gate_up_proj for better coverage)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "gate_up_proj"],  # 2025 canonical list
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Disable cache for training (generation uses cache, training shouldn't)
    # This avoids warnings and saves memory
    model.config.use_cache = False
    
    model.print_trainable_parameters()

    # torch.compile() is NOT compatible with quantized models + PEFT, even in transformers 4.45.0
    # The error message is clear: "You cannot fine-tune quantized model with torch.compile()"
    # 4-bit QLoRA is still fast due to memory efficiency and allows larger batch sizes
    # print("Applying torch.compile() – expect +35–45% speedup...")
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    print("Loading datasets...")
    train_dataset = load_jsonl(args.train_file)
    if args.val_file:
        val_dataset = load_jsonl(args.val_file)
        print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} examples")
    else:
        val_dataset = None
        print(f"Train: {len(train_dataset):,} examples (no validation dataset)")

    print("Formatting + tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: format_messages(x, tokenizer),
        batched=True, num_proc=4, desc="Formatting train"
    ).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=args.max_seq_length),
        batched=True, num_proc=4, remove_columns=["messages", "text"], desc="Tokenizing train"
    )

    if val_dataset is not None:
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
        per_device_eval_batch_size=2,           # Reduced for safety during eval (prevents OOM)
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,

        bf16=True,
        gradient_checkpointing=True,            # Enabled to save memory
        optim="adamw_torch",                    # Standard optimizer (8-bit optimizers have compatibility issues with accelerate in 4.45.0)
                                                # adamw_torch is stable and works well with 4-bit QLoRA
        logging_steps=10,
        save_steps=5000,                       # Less frequent checkpoints for speed
        save_total_limit=3,

        # Validation – smart & low overhead (only if val_file provided)
        eval_strategy="steps" if val_dataset is not None else "no",  # Updated from evaluation_strategy (deprecated in 4.46+)
        eval_steps=5000 if val_dataset is not None else None,  # Less frequent validation for speed
        eval_accumulation_steps=16 if val_dataset is not None else None,  # Increased to prevent OOM during eval (processes in smaller chunks)
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,

        # Speed flags (reduced for memory efficiency with VLLM on GPU 1)
        dataloader_num_workers=12,             # Reduced from 12 to save memory
        dataloader_pin_memory=True,            # Faster GPU transfer
        dataloader_prefetch_factor=4,         # Reduced from 4 to save memory
        torch_compile=False,                    # Using manual torch.compile() above instead

        report_to="none",
        disable_tqdm=False,
    )

    # Prepare callbacks - Early Stopping only if validation is enabled
    callbacks = []
    if val_dataset is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=1,   # Stop if next eval doesn't improve loss (since eval every 5000 steps)
            early_stopping_threshold=0.0  # Any improvement counts
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks if callbacks else None,
    )

    print("\nStarting training – this will be FAST and HIGH-QUALITY!\n")
    trainer.train()

    # Save final model
    # If validation was used and load_best_model_at_end=True, best model is already loaded by trainer
    if val_dataset is not None and training_args.load_best_model_at_end:
        best_dir = os.path.join(args.output_dir, "final-best")
    else:
        best_dir = os.path.join(args.output_dir, "final")
    
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    
    if val_dataset is not None and training_args.load_best_model_at_end:
        print(f"\nTraining complete! Best model (based on eval_loss) saved to: {best_dir}")
    else:
        print(f"\nTraining complete! Final model saved to: {best_dir}")

if __name__ == "__main__":
    main()