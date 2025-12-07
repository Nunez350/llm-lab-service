#!/usr/bin/env python
"""
Dataset Preparation Script for High-Quality LoRA Training

Merges OpenHermes 2.5 and SlimOrca datasets with:
- English filtering
- Deduplication
- Length normalization
- Quality filtering
- Train/val split

Usage:
    python prepare_dataset.py
"""

import hashlib
import random
import json
from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets

# ========== CONFIG ==========
OPENHERMES_NAME = "teknium/OpenHermes-2.5"
SLIMORCA_NAME = "Open-Orca/SlimOrca"

# Target sizes (adjust as needed)
TARGET_OH = 300_000      # OpenHermes samples
TARGET_SO = 150_000      # SlimOrca samples

# Quality filters
MAX_CHARS_PER_CONVO = 6000
MIN_CHARS_PER_CONVO = 50
MIN_TURNS = 2

# Output files
OUTPUT_DIR = Path("./datasets")
TRAIN_FILE = OUTPUT_DIR / "merged_train.jsonl"
VAL_FILE = OUTPUT_DIR / "merged_val.jsonl"
STATS_FILE = OUTPUT_DIR / "dataset_stats.json"

# Split ratio
VAL_RATIO = 0.1

RANDOM_SEED = 42
# ============================


def normalize_conversations(example):
    """
    Normalize conversations to standard messages format:
    [{'role': 'user'/'assistant'/'system', 'content': text}, ...]
    """
    conv = example.get("conversations")
    if conv is None:
        return {"messages": []}

    messages = []
    for turn in conv:
        src_role = turn.get("from", "").lower()
        text = (turn.get("value") or "").strip()
        if not text:
            continue

        # Map source role to standard role
        if src_role in ["human", "user"]:
            role = "user"
        elif src_role in ["gpt", "assistant"]:
            role = "assistant"
        elif src_role == "system":
            role = "system"
        else:
            # Fallback: treat unknown as user
            role = "user"

        messages.append({"role": role, "content": text})

    return {"messages": messages}


def filter_quality(example):
    """
    Quality filters:
    - Must have user + assistant turns
    - Character count within bounds
    - Minimum turns requirement
    """
    msgs = example.get("messages") or []

    # Must have minimum turns
    if len(msgs) < MIN_TURNS:
        return False

    # Must have both user and assistant
    has_user = any(m["role"] == "user" for m in msgs)
    has_assistant = any(m["role"] == "assistant" for m in msgs)
    if not (has_user and has_assistant):
        return False

    # Check character count
    total_chars = sum(len(m["content"]) for m in msgs)
    if total_chars < MIN_CHARS_PER_CONVO or total_chars > MAX_CHARS_PER_CONVO:
        return False

    # Check for empty content
    if any(not m["content"].strip() for m in msgs):
        return False

    return True


def hash_example(example):
    """
    Create hash of conversation content for deduplication.
    """
    msgs = example.get("messages") or []
    text = "\n".join(f"{m['role']}:{m['content']}" for m in msgs)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_stats(dataset, name):
    """
    Calculate dataset statistics.
    """
    total_examples = len(dataset)

    total_chars = 0
    total_turns = 0
    role_counts = {"user": 0, "assistant": 0, "system": 0}

    for ex in dataset:
        msgs = ex.get("messages") or []
        total_turns += len(msgs)
        for msg in msgs:
            total_chars += len(msg["content"])
            role_counts[msg["role"]] = role_counts.get(msg["role"], 0) + 1

    avg_chars = total_chars / total_examples if total_examples > 0 else 0
    avg_turns = total_turns / total_examples if total_examples > 0 else 0

    return {
        "name": name,
        "examples": total_examples,
        "total_chars": total_chars,
        "avg_chars_per_example": round(avg_chars, 2),
        "avg_turns_per_example": round(avg_turns, 2),
        "role_distribution": role_counts
    }


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    stats = {}

    # ===== LOAD OPENHERMES =====
    print("\n" + "="*60)
    print("Loading OpenHermes 2.5...")
    print("="*60)
    oh = load_dataset(OPENHERMES_NAME, split="train", streaming=False)
    print(f"OpenHermes loaded: {len(oh):,} examples")

    # Normalize format
    print("Normalizing conversations...")
    oh_normalized = oh.map(
        normalize_conversations,
        remove_columns=[c for c in oh.column_names if c != "conversations"],
        desc="Normalizing OpenHermes"
    )

    # Apply quality filters
    print("Applying quality filters...")
    oh_filtered = oh_normalized.filter(filter_quality, desc="Filtering OpenHermes")
    if len(oh_normalized) > 0:
        print(f"After quality filter: {len(oh_filtered):,} examples ({len(oh_filtered)/len(oh_normalized)*100:.1f}% kept)")
    else:
        print(f"After quality filter: {len(oh_filtered):,} examples")

    # Sample to target size
    if TARGET_OH and len(oh_filtered) > TARGET_OH:
        print(f"Sampling to {TARGET_OH:,} examples...")
        oh_filtered = oh_filtered.shuffle(seed=RANDOM_SEED).select(range(TARGET_OH))

    stats['openhermes'] = calculate_stats(oh_filtered, "OpenHermes 2.5")
    print(f"Final OpenHermes: {len(oh_filtered):,} examples")

    # ===== LOAD SLIMORCA =====
    print("\n" + "="*60)
    print("Loading SlimOrca...")
    print("="*60)
    so = load_dataset(SLIMORCA_NAME, split="train", streaming=False)
    print(f"SlimOrca loaded: {len(so):,} examples")

    # Normalize format
    print("Normalizing conversations...")
    so_normalized = so.map(
        normalize_conversations,
        remove_columns=[c for c in so.column_names if c != "conversations"],
        desc="Normalizing SlimOrca"
    )

    # Apply quality filters
    print("Applying quality filters...")
    so_filtered = so_normalized.filter(filter_quality, desc="Filtering SlimOrca")
    if len(so_normalized) > 0:
        print(f"After quality filter: {len(so_filtered):,} examples ({len(so_filtered)/len(so_normalized)*100:.1f}% kept)")
    else:
        print(f"After quality filter: {len(so_filtered):,} examples")

    # Sample to target size
    if TARGET_SO and len(so_filtered) > TARGET_SO:
        print(f"Sampling to {TARGET_SO:,} examples...")
        so_filtered = so_filtered.shuffle(seed=RANDOM_SEED).select(range(TARGET_SO))

    stats['slimorca'] = calculate_stats(so_filtered, "SlimOrca")
    print(f"Final SlimOrca: {len(so_filtered):,} examples")

    # ===== MERGE DATASETS =====
    print("\n" + "="*60)
    print("Merging datasets...")
    print("="*60)
    merged = concatenate_datasets([oh_filtered, so_filtered])
    merged = merged.shuffle(seed=RANDOM_SEED)
    print(f"Merged size before dedup: {len(merged):,} examples")

    # ===== DEDUPLICATION =====
    print("\nDeduplicating...")
    seen = set()
    keep_indices = []

    for i, ex in enumerate(merged):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(merged):,} examples...")

        h = hash_example(ex)
        if h in seen:
            continue
        seen.add(h)
        keep_indices.append(i)

    merged = merged.select(keep_indices)
    duplicates_removed = len(seen) - len(merged)
    print(f"Removed {duplicates_removed:,} duplicates")
    print(f"Final merged size: {len(merged):,} examples")

    stats['merged'] = calculate_stats(merged, "Merged Dataset")

    # ===== TRAIN/VAL SPLIT =====
    print("\n" + "="*60)
    print("Creating train/validation split...")
    print("="*60)

    val_size = int(len(merged) * VAL_RATIO)
    train_size = len(merged) - val_size

    train_data = merged.select(range(train_size))
    val_data = merged.select(range(train_size, len(merged)))

    print(f"Train: {len(train_data):,} examples ({(1-VAL_RATIO)*100:.0f}%)")
    print(f"Val:   {len(val_data):,} examples ({VAL_RATIO*100:.0f}%)")

    stats['train'] = calculate_stats(train_data, "Training Set")
    stats['val'] = calculate_stats(val_data, "Validation Set")

    # ===== SAVE TO JSONL =====
    print("\n" + "="*60)
    print("Saving datasets...")
    print("="*60)

    # Save training data
    print(f"Writing {TRAIN_FILE}...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    # Save validation data
    print(f"Writing {VAL_FILE}...")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    # Save statistics
    print(f"Writing {STATS_FILE}...")
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nDataset Statistics:")
    print(f"  OpenHermes:  {stats['openhermes']['examples']:>8,} examples")
    print(f"  SlimOrca:    {stats['slimorca']['examples']:>8,} examples")
    print(f"  Merged:      {stats['merged']['examples']:>8,} examples")
    print(f"  Duplicates:  {duplicates_removed:>8,} removed")
    print(f"\nFinal Split:")
    print(f"  Training:    {stats['train']['examples']:>8,} examples")
    print(f"  Validation:  {stats['val']['examples']:>8,} examples")
    print(f"\nQuality Metrics:")
    print(f"  Avg chars/example:  {stats['merged']['avg_chars_per_example']:.0f}")
    print(f"  Avg turns/example:  {stats['merged']['avg_turns_per_example']:.1f}")
    print(f"\nOutput Files:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {STATS_FILE}")
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
