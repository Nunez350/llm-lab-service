#!/usr/bin/env python
"""
Monitor training progress - Extract loss, gradient norms, and sequence lengths from logs

Usage:
    python monitor_training.py [--log_file PATH] [--watch]
"""

import argparse
import re
import json
from pathlib import Path

def parse_time_str(time_str):
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def parse_training_log(log_file):
    metrics = {
        'steps': [],
        'loss': [],
        'eval_loss': [],
        'learning_rate': [],
        'current_step': 0,
        'total_steps': 17190,
        'time_per_step': None,
        'eta_seconds': None,
        'progress_percent': 0.0
    }
    
    if not Path(log_file).exists():
        return metrics
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return metrics
    
    for line in reversed(lines):
        if '{' in line and 'loss' in line.lower():
            try:
                json_start = line.find('{')
                json_end = line.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = line[json_start:json_end]
                    json_str = json_str.replace("'", '"')
                    data = json.loads(json_str)
                    
                    if 'step' in data and data['step'] not in metrics['steps']:
                        metrics['steps'].append(data['step'])
                    if 'loss' in data and 'eval_loss' not in str(data.get('loss', '')):
                        metrics['loss'].append((data.get('step', 0), data['loss']))
                    if 'eval_loss' in data:
                        metrics['eval_loss'].append((data.get('step', 0), data['eval_loss']))
                    if 'learning_rate' in data:
                        metrics['learning_rate'].append((data.get('step', 0), data['learning_rate']))
            except (json.JSONDecodeError, KeyError):
                pass
        
        progress_match = re.search(r'(\d+)%\|.*?\| (\d+)/(\d+) \[([\d:]+)<([\d:]+), ([\d.]+)s/it', line)
        if progress_match:
            percent = int(progress_match.group(1))
            current_step = int(progress_match.group(2))
            total_steps = int(progress_match.group(3))
            eta = parse_time_str(progress_match.group(5))
            time_per_step = float(progress_match.group(6))
            
            metrics['current_step'] = current_step
            metrics['total_steps'] = total_steps
            metrics['progress_percent'] = percent
            metrics['time_per_step'] = time_per_step
            metrics['eta_seconds'] = eta
            break
    
    metrics['loss'].sort(key=lambda x: x[0])
    metrics['eval_loss'].sort(key=lambda x: x[0])
    metrics['learning_rate'].sort(key=lambda x: x[0])
    metrics['steps'].sort()
    
    return metrics

def print_metrics(metrics):
    print("\n" + "="*70)
    print("📊 Training Progress Monitor")
    print("="*70)
    
    if metrics['current_step'] > 0:
        print(f"\n📍 Current Step: {metrics['current_step']:,} / {metrics['total_steps']:,}")
        print(f"   Progress: {metrics['progress_percent']:.1f}%")
        remaining = metrics['total_steps'] - metrics['current_step']
        print(f"   Remaining: {remaining:,} steps")
    
    if metrics['time_per_step']:
        print(f"\n⏱️  Time per step: {metrics['time_per_step']:.2f}s")
        if metrics['eta_seconds']:
            print(f"   ETA: {format_time(metrics['eta_seconds'])}")
        
        if metrics['current_step'] > 0 and metrics['time_per_step']:
            remaining = metrics['total_steps'] - metrics['current_step']
            total_remaining = remaining * metrics['time_per_step']
            print(f"   Estimated remaining: {format_time(int(total_remaining))}")
    
    if metrics['loss']:
        latest_loss = metrics['loss'][-1]
        print(f"\n📉 Training Loss:")
        print(f"   Latest (step {latest_loss[0]}): {latest_loss[1]:.4f}")
        
        if len(metrics['loss']) > 1:
            first_loss = metrics['loss'][0]
            change = latest_loss[1] - first_loss[1]
            trend = "↓ Decreasing" if change < 0 else "↑ Increasing"
            print(f"   First (step {first_loss[0]}): {first_loss[1]:.4f}")
            print(f"   Change: {change:+.4f} ({trend})")
            
            if len(metrics['loss']) >= 5:
                recent = [l[1] for l in metrics['loss'][-5:]]
                recent_change = recent[-1] - recent[0]
                print(f"   Recent trend (last 5): {recent_change:+.4f}")
    
    if metrics['eval_loss']:
        latest_eval = metrics['eval_loss'][-1]
        print(f"\n📊 Evaluation Loss:")
        print(f"   Latest (step {latest_eval[0]}): {latest_eval[1]:.4f}")
        
        if len(metrics['eval_loss']) > 1:
            first_eval = metrics['eval_loss'][0]
            change = latest_eval[1] - first_eval[1]
            trend = "↓ Decreasing" if change < 0 else "↑ Increasing"
            print(f"   First (step {first_eval[0]}): {first_eval[1]:.4f}")
            print(f"   Change: {change:+.4f} ({trend})")
    
    if metrics['learning_rate']:
        latest_lr = metrics['learning_rate'][-1]
        print(f"\n📈 Learning Rate:")
        print(f"   Current: {latest_lr[1]:.6f}")
    
    print(f"\n💡 Note: Gradient norms and sequence lengths are not logged by default.")
    print(f"   To add them, modify train_custom_dataset.py with a custom callback.")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Monitor training progress from logs')
    parser.add_argument('--log_file', type=str,
                        default='/home/rnu/mnt/models/unsloth/phi3-finetuned-english-only-bf16/training.log',
                        help='Path to training log file')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously monitor (updates every 5 seconds)')
    
    args = parser.parse_args()
    
    if args.watch:
        import time
        try:
            while True:
                print("\033[2J\033[H", end="")
                metrics = parse_training_log(args.log_file)
                print_metrics(metrics)
                print("Press Ctrl+C to stop monitoring...")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        metrics = parse_training_log(args.log_file)
        print_metrics(metrics)

if __name__ == "__main__":
    main()
