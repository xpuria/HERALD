"""
Plot Model Comparison: LSTM vs Transformer vs HRM

Parse training logs and create comparison plots for loss and accuracy.
"""

import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def parse_log_file(log_path, model_name):
    """Parse training log and extract epoch, loss, and accuracy."""
    epochs = []
    losses = []
    accuracies = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Pattern 1: "Epoch 0 - LR: 4.97e-04 - Loss: 0.7004 - Acc: 0.480"
            match = re.search(r'Epoch (\d+).*Loss:\s*([\d.]+).*Acc:\s*([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc * 100)  # Convert to percentage
                continue
            
            # Pattern 2 (HRM): "Epoch 0 completed: {'total_loss': 0.7062, ..., 'accuracy': 0.5272, ...}"
            match = re.search(r"Epoch (\d+) completed:.*'total_loss':\s*([\d.]+).*'accuracy':\s*([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc * 100)  # Convert to percentage
    
    return epochs, losses, accuracies


def create_comparison_plots(data_dict, output_dir='.'):
    """Create comparison plots for all models."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'LSTM': '#1f77b4',      # Blue
        'Transformer': '#ff7f0e',  # Orange
        'HRM': '#2ca02c'         # Green
    }
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss over epochs
    for model_name, (epochs, losses, _) in data_dict.items():
        ax1.plot(epochs, losses, 
                marker='o', 
                linewidth=2, 
                markersize=4,
                color=colors.get(model_name, 'gray'),
                label=f'{model_name} (Final: {losses[-1]:.4f})',
                alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Plot 2: Accuracy over epochs
    for model_name, (epochs, _, accuracies) in data_dict.items():
        ax2.plot(epochs, accuracies, 
                marker='s', 
                linewidth=2, 
                markersize=4,
                color=colors.get(model_name, 'gray'),
                label=f'{model_name} (Final: {accuracies[-1]:.1f}%)',
                alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Training Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'model_comparison_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {output_path}")
    plt.close()
    
    # Create individual comparison table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Epochs':<10} {'Final Loss':<15} {'Final Acc':<15} {'Best Acc':<15}")
    print("-"*70)
    
    for model_name, (epochs, losses, accuracies) in data_dict.items():
        final_loss = losses[-1]
        final_acc = accuracies[-1]
        best_acc = max(accuracies)
        n_epochs = len(epochs)
        
        print(f"{model_name:<15} {n_epochs:<10} {final_loss:<15.4f} {final_acc:<15.1f}% {best_acc:<15.1f}%")
    
    print("="*70)


def main():
    # Define log file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    log_files = {
        'LSTM': os.path.join(base_dir, 'checkpoints_lstm_20epochs/train_lstm_baseline.log'),
        'Transformer': os.path.join(base_dir, 'checkpoints_transformer/train_transformer_baseline.log'),
        'HRM': os.path.join(base_dir, 'checkpoints_hrm_20epochs/train_working_hrm.log'),
    }
    
    # Parse all log files
    data_dict = {}
    
    print("Parsing training logs...")
    for model_name, log_path in log_files.items():
        if os.path.exists(log_path):
            print(f"  ✓ Loading {model_name} from {log_path}")
            epochs, losses, accuracies = parse_log_file(log_path, model_name)
            
            if epochs:
                data_dict[model_name] = (epochs, losses, accuracies)
                print(f"    - Found {len(epochs)} epochs")
            else:
                print(f"    ⚠ No data found in log file")
        else:
            print(f"  ✗ Log file not found: {log_path}")
    
    if not data_dict:
        print("\n❌ No data found to plot!")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(data_dict)} models\n")
    
    # Create plots
    create_comparison_plots(data_dict, output_dir=base_dir)
    
    # Performance analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Find best model
    best_model = None
    best_acc = 0
    
    for model_name, (_, _, accuracies) in data_dict.items():
        max_acc = max(accuracies)
        if max_acc > best_acc:
            best_acc = max_acc
            best_model = model_name
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Best Accuracy: {best_acc:.1f}%")
    
    # Calculate improvements
    if 'HRM' in data_dict and 'Transformer' in data_dict:
        hrm_acc = max(data_dict['HRM'][2])
        trans_acc = max(data_dict['Transformer'][2])
        improvement = hrm_acc - trans_acc
        print(f"\n📊 HRM vs Transformer:")
        print(f"   HRM Accuracy: {hrm_acc:.1f}%")
        print(f"   Transformer Accuracy: {trans_acc:.1f}%")
        print(f"   Improvement: +{improvement:.1f}%")
    
    if 'HRM' in data_dict and 'LSTM' in data_dict:
        hrm_acc = max(data_dict['HRM'][2])
        lstm_acc = max(data_dict['LSTM'][2])
        improvement = hrm_acc - lstm_acc
        print(f"\n📊 HRM vs LSTM:")
        print(f"   HRM Accuracy: {hrm_acc:.1f}%")
        print(f"   LSTM Accuracy: {lstm_acc:.1f}%")
        print(f"   Improvement: +{improvement:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

