"""
Model Comparison Script

Compare LSTM, Transformer, and HRM models on the same dataset.
Generates comprehensive metrics and visualizations.
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.lstm_baseline import create_lstm_baseline
from src.models.transformer_baseline import create_transformer_baseline
from src.models.hrm import HierarchicalReasoningModel_ACTV1
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed
import numpy as np
from tqdm import tqdm
import json


def load_model(model_type, checkpoint_path, device, config=None):
    """Load a trained model from checkpoint"""
    
    if model_type == "lstm":
        model = create_lstm_baseline(
            vocab_size=512,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 4),
            dropout=0.1,
            seq_len=16,
            use_attention=config.get('use_attention', False)
        ).to(device)
        
    elif model_type == "transformer":
        model = create_transformer_baseline(
            vocab_size=512,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 4),
            ff_dim=config.get('hidden_size', 128) * 4,
            dropout=0.1,
            seq_len=16,
            use_causal_mask=True
        ).to(device)
        
    elif model_type == "hrm":
        hrm_config = {
            'batch_size': config.get('batch_size', 8),
            'seq_len': 16,
            'vocab_size': 512,
            'H_cycles': 2,
            'L_cycles': 4,
            'H_layers': 2,
            'L_layers': 2,
            'hidden_size': config.get('hidden_size', 128),
            'expansion': 4.0,
            'num_heads': 4,
            'halt_max_steps': 1,
            'halt_exploration_prob': 0.0,
            'forward_dtype': 'float32'
        }
        model = HierarchicalReasoningModel_ACTV1(hrm_config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(model, tokens, labels, device, seq_len=16, model_type="lstm", batch_size=8):
    """Evaluate a model on the test set"""
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Split into segments
    stream_len = tokens.shape[0]
    num_segments = stream_len // seq_len
    
    # For HRM, we need to manage carry state
    carry = None
    
    with torch.no_grad():
        for i in tqdm(range(num_segments), desc=f"Evaluating {model_type.upper()}"):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[start_idx:end_idx].unsqueeze(0).to(device)
            target_labels = labels[start_idx:end_idx].unsqueeze(0).to(device)
            
            try:
                # Forward pass
                if model_type == "hrm":
                    # HRM requires different interface
                    batch = {
                        "inputs": input_tokens,
                        "labels": target_labels
                    }
                    if carry is None or i % 20 == 0:
                        carry = model.initial_carry(batch)
                    
                    new_carry, outputs = model(carry=carry, batch=batch)
                    carry = new_carry
                else:
                    # LSTM/Transformer
                    outputs = model(input_tokens)
                
                # Extract logits
                logits = outputs["logits"]  # (1, seq, 2)
                logits_flat = logits.reshape(-1, 2)
                labels_flat = target_labels.reshape(-1)
                
                # Compute loss
                loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    
                    # Compute accuracy
                    preds = torch.argmax(logits_flat, dim=-1)
                    acc = (preds == labels_flat).float().mean()
                    total_acc += acc.item()
                    
                    # Store predictions and probabilities
                    probs = F.softmax(logits_flat, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels_flat.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
                    
                    total_samples += 1
                    
            except Exception as e:
                print(f"Error in segment {i}: {e}")
                if model_type == "hrm":
                    carry = None
                continue
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'loss': total_loss / total_samples if total_samples > 0 else float('inf'),
        'accuracy': total_acc / total_samples if total_samples > 0 else 0.0,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'total_samples': total_samples
    }
    
    # Additional metrics
    if len(all_labels) > 0:
        # Precision, Recall, F1
        true_pos = np.sum((all_preds == 1) & (all_labels == 1))
        false_pos = np.sum((all_preds == 1) & (all_labels == 0))
        false_neg = np.sum((all_preds == 0) & (all_labels == 1))
        true_neg = np.sum((all_preds == 0) & (all_labels == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['true_pos'] = int(true_pos)
        metrics['false_pos'] = int(false_pos)
        metrics['true_neg'] = int(true_neg)
        metrics['false_neg'] = int(false_neg)
    
    return metrics


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = VQVAE(
        num_hiddens=64,
        num_residual_layers=2,
        num_residual_hiddens=32,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    ).to(device)
    
    if args.vqvae_checkpoint:
        vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
        print(f"Loaded VQ-VAE from {args.vqvae_checkpoint}\n")
    
    # Prepare data
    print("Preparing data...")
    dataset = HRMDataset(
        csv_path=args.data_path,
        vqvae_model=vqvae,
        seq_len=64,
        device=str(device),
        target_col=args.target_col
    )
    
    tokens, labels = dataset.get_all_tokens_and_labels()
    print(f"Total tokens: {tokens.shape[0]}\n")
    
    # Model configurations
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'batch_size': args.batch_size
    }
    
    results = {}
    
    # Evaluate each model
    model_configs = [
        ("lstm", args.lstm_checkpoint),
        ("transformer", args.transformer_checkpoint),
        ("hrm", args.hrm_checkpoint)
    ]
    
    for model_type, checkpoint_path in model_configs:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"Evaluating {model_type.upper()} Model")
            print(f"{'='*60}")
            
            try:
                # Load model
                model = load_model(model_type, checkpoint_path, device, config)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"Parameters: {param_count:,}")
                
                # Evaluate
                metrics = evaluate_model(
                    model, tokens, labels, device, 
                    seq_len=args.seq_len, 
                    model_type=model_type,
                    batch_size=args.batch_size
                )
                
                # Store results (exclude large arrays for JSON)
                results[model_type] = {
                    'parameters': param_count,
                    'loss': metrics['loss'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1': metrics.get('f1', 0.0),
                    'total_samples': metrics['total_samples']
                }
                
                # Print results
                print(f"\nResults:")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
                print(f"  Recall: {metrics.get('recall', 0.0):.4f}")
                print(f"  F1 Score: {metrics.get('f1', 0.0):.4f}")
                print(f"  Samples: {metrics['total_samples']}")
                
                if 'true_pos' in metrics:
                    print(f"\nConfusion Matrix:")
                    print(f"  True Pos:  {metrics['true_pos']}")
                    print(f"  False Pos: {metrics['false_pos']}")
                    print(f"  True Neg:  {metrics['true_neg']}")
                    print(f"  False Neg: {metrics['false_neg']}")
                
            except Exception as e:
                print(f"Error evaluating {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        else:
            print(f"\nSkipping {model_type.upper()} - checkpoint not found: {checkpoint_path}")
    
    # Save comparison results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    if results:
        # Create comparison table
        print(f"{'Model':<15} {'Params':<12} {'Loss':<10} {'Acc':<10} {'F1':<10}")
        print("-" * 60)
        for model_type in ['lstm', 'transformer', 'hrm']:
            if model_type in results and 'error' not in results[model_type]:
                r = results[model_type]
                params_str = f"{r['parameters']:,}"
                print(f"{model_type.upper():<15} {params_str:<12} {r['loss']:<10.4f} "
                      f"{r['accuracy']:<10.4f} {r['f1']:<10.4f}")
        
        # Save to file
        output_path = os.path.join(args.output_dir, "model_comparison_results.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Save text summary
        text_path = os.path.join(args.output_dir, "model_comparison_results.txt")
        with open(text_path, 'w') as f:
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Model':<15} {'Params':<12} {'Loss':<10} {'Acc':<10} {'F1':<10}\n")
            f.write("-" * 60 + "\n")
            for model_type in ['lstm', 'transformer', 'hrm']:
                if model_type in results and 'error' not in results[model_type]:
                    r = results[model_type]
                    params_str = f"{r['parameters']:,}"
                    f.write(f"{model_type.upper():<15} {params_str:<12} {r['loss']:<10.4f} "
                           f"{r['accuracy']:<10.4f} {r['f1']:<10.4f}\n")
        
        print(f"Text summary saved to: {text_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LSTM, Transformer, and HRM models")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CSV data file")
    parser.add_argument("--target_col", type=str, default="Close",
                       help="Target column name")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="Path to VQ-VAE checkpoint")
    
    # Model checkpoint paths
    parser.add_argument("--lstm_checkpoint", type=str, default=None,
                       help="Path to LSTM checkpoint")
    parser.add_argument("--transformer_checkpoint", type=str, default=None,
                       help="Path to Transformer checkpoint")
    parser.add_argument("--hrm_checkpoint", type=str, default=None,
                       help="Path to HRM checkpoint")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--seq_len", type=int, default=16,
                       help="Sequence length")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    main(args)
