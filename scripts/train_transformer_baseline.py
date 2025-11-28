"""
Training script for Transformer Baseline on Financial Time Series

This script trains a Transformer model using the same setup as LSTM/HRM for fair comparison.
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.transformer_baseline import create_transformer_baseline
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging
from tqdm import tqdm
import logging


def train_epoch(model, tokens, labels, optimizer, device, epoch, seq_len=16):
    """Training loop for one epoch"""
    model.train()
    
    batch_size, total_steps = tokens.shape
    num_segments = total_steps // seq_len
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    valid_batches = 0
    
    progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
    
    for i in progress_bar:
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        input_tokens = tokens[:, start_idx:end_idx].to(device)
        target_labels = labels[:, start_idx:end_idx].to(device)
        
        try:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_tokens)
            
            # Cross-entropy loss
            logits = outputs["logits"]  # (batch, seq, 2)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = target_labels.reshape(-1)
            
            loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
            
            # Skip if NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected, skipping batch {i}")
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Skip extreme gradients
            if grad_norm > 100.0:
                logging.warning(f"Extreme gradient: {grad_norm:.2f}, skipping")
                continue
            
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                preds = torch.argmax(logits_flat, dim=-1)
                acc = (preds == labels_flat).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            valid_batches += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc.item():.3f}",
                "grad": f"{grad_norm:.2f}"
            })
            
        except Exception as e:
            logging.error(f"Error in batch {i}: {e}")
            continue
    
    # Average metrics
    if valid_batches > 0:
        epoch_loss /= valid_batches
        epoch_acc /= valid_batches
    
    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "valid_batches": valid_batches
    }


def main(args):
    set_seed(args.seed)
    setup_logging(os.path.join(args.checkpoint_dir, "train_transformer_baseline.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load VQ-VAE
    logging.info("Loading VQ-VAE...")
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
        logging.info(f"Loaded VQ-VAE from {args.vqvae_checkpoint}")
    
    # Prepare data
    logging.info("Preparing data...")
    dataset = HRMDataset(
        csv_path=args.data_path,
        vqvae_model=vqvae,
        seq_len=64,
        device=str(device),
        target_col=args.target_col
    )
    
    tokens, labels = dataset.get_all_tokens_and_labels()
    logging.info(f"Total tokens: {tokens.shape[0]}")
    
    # Reshape for batching
    batch_size = args.batch_size
    stream_len = tokens.shape[0] // batch_size
    
    tokens = tokens[:batch_size * stream_len].reshape(batch_size, stream_len)
    labels = labels[:batch_size * stream_len].reshape(batch_size, stream_len)
    
    # Create Transformer model
    logging.info("Creating Transformer model...")
    model = create_transformer_baseline(
        vocab_size=512,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.hidden_size * 4,
        dropout=args.dropout,
        seq_len=args.seq_len,
        use_causal_mask=True,
        use_relative_positions=args.use_relative_positions
    ).to(device)
    
    # Count parameters
    total_params = model.count_parameters()
    logging.info(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # Training loop
    logging.info("Starting training...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = train_epoch(
            model, tokens, labels, optimizer, device, epoch, args.seq_len
        )
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': {
                'vocab_size': 512,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'seq_len': args.seq_len
            }
        }
        
        # Save epoch checkpoint
        torch.save(
            checkpoint,
            os.path.join(args.checkpoint_dir, f"transformer_baseline_epoch_{epoch}.pt")
        )
        
        # Save best checkpoint
        if metrics["loss"] < best_loss and metrics["valid_batches"] > 0:
            best_loss = metrics["loss"]
            torch.save(
                checkpoint,
                os.path.join(args.checkpoint_dir, "best_transformer_baseline.pt")
            )
        
        logging.info(
            f"Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.2e} - "
            f"Loss: {metrics['loss']:.4f} - Acc: {metrics['accuracy']:.3f} - "
            f"Valid: {metrics['valid_batches']} - Best: {best_loss:.4f}"
        )
    
    logging.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Baseline")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to CSV data file")
    parser.add_argument("--target_col", type=str, default="Close",
                       help="Target column name")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="Path to VQ-VAE checkpoint")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability")
    parser.add_argument("--use_relative_positions", action="store_true",
                       help="Use learned relative positional embeddings")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--seq_len", type=int, default=16,
                       help="Sequence length")
    parser.add_argument("--epochs", type=int, default=40,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Other arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_transformer",
                       help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    main(args)

