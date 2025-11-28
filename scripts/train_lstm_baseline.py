"""
Train LSTM Baseline Model

This script trains an LSTM model with the same hyperparameters as HRM
to provide a fair comparison baseline.
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.lstm_baseline import create_lstm_baseline
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging
from tqdm import tqdm
import logging


class LSTMTrainer:
    """Trainer for LSTM baseline model"""
    
    def __init__(self, model, optimizer, device, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
    def compute_loss(self, outputs, labels):
        """Compute classification loss and metrics"""
        logits = outputs["logits"]  # (batch_size, seq_len, 2)
        
        # Flatten for loss computation
        logits_flat = logits.reshape(-1, 2)
        labels_flat = labels.reshape(-1)
        
        # Cross-entropy loss with label smoothing
        classification_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            label_smoothing=0.1,
            reduction='mean'
        )
        
        # Compute accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=1)
            accuracy = (predictions == labels_flat).float().mean()
        
        return {
            "loss": classification_loss,
            "accuracy": accuracy
        }
    
    def train_epoch(self, tokens, labels, epoch):
        """Train one epoch"""
        self.model.train()
        
        batch_size, total_steps = tokens.shape
        seq_len = self.model.config.seq_len
        num_segments = total_steps // seq_len
        
        # Initialize LSTM hidden state
        hidden_state = None
        
        epoch_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "valid_batches": 0
        }
        
        progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
        
        for i in progress_bar:
            # Get segment
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(self.device)
            target_labels = labels[:, start_idx:end_idx].to(self.device)
            
            # Reset state periodically for stability
            if i % 20 == 0:
                hidden_state = None
            
            try:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_tokens, hidden_state)
                hidden_state = outputs.get("hidden_state", None)
                
                # Detach hidden state to prevent backprop through time issues
                if hidden_state is not None:
                    hidden_state = tuple(h.detach() for h in hidden_state)
                
                # Compute loss
                loss_dict = self.compute_loss(outputs, target_labels)
                loss = loss_dict["loss"]
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                    logging.warning(f"Unstable loss detected: {loss:.2e}, resetting state")
                    hidden_state = None
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=0.5
                )
                
                if grad_norm > 5.0:
                    logging.warning(f"Large gradient norm: {grad_norm:.2f}, skipping update")
                    hidden_state = None
                    continue
                
                self.optimizer.step()
                
                # Update metrics
                for key in loss_dict:
                    if key in epoch_metrics:
                        epoch_metrics[key] += loss_dict[key].item()
                epoch_metrics["valid_batches"] += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{loss_dict['accuracy'].item():.3f}",
                    "grad_norm": f"{grad_norm:.2f}"
                })
                
            except Exception as e:
                logging.error(f"Error in batch {i}: {e}")
                hidden_state = None
                continue
        
        # Compute averages
        if epoch_metrics["valid_batches"] > 0:
            for key in epoch_metrics:
                if key != "valid_batches":
                    epoch_metrics[key] /= epoch_metrics["valid_batches"]
        
        logging.info(f"Epoch {epoch} completed: {epoch_metrics}")
        
        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics': epoch_metrics
        }, os.path.join(self.checkpoint_dir, f"lstm_baseline_epoch_{epoch}.pt"))
        
        return epoch_metrics


def main(args):
    set_seed(args.seed)
    setup_logging(os.path.join(args.checkpoint_dir, "train_lstm_baseline.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 1. Load VQ-VAE
    logging.info("Loading VQ-VAE...")
    vqvae = VQVAE(
        num_hiddens=args.vq_hiddens,
        num_residual_layers=args.vq_res_layers,
        num_residual_hiddens=args.vq_res_hiddens,
        num_embeddings=args.vocab_size,
        embedding_dim=args.vq_emb_dim,
        commitment_cost=0.25
    ).to(device)
    
    if args.vqvae_checkpoint:
        vqvae.load_state_dict(torch.load(args.vqvae_checkpoint, map_location=device))
        logging.info(f"Loaded VQ-VAE from {args.vqvae_checkpoint}")
    
    # 2. Prepare Data
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
    
    # 3. Initialize LSTM Model
    logging.info("Initializing LSTM Baseline Model...")
    
    # Match HRM layer count: L_layers + H_layers
    total_lstm_layers = args.l_layers + args.h_layers
    
    model = create_lstm_baseline(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=total_lstm_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
        use_attention=args.use_attention,
        bidirectional=args.bidirectional
    ).to(device)
    
    # Count parameters
    total_params = model.count_parameters()
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Configuration:")
    logging.info(f"  - Vocab size: {args.vocab_size}")
    logging.info(f"  - Hidden size: {args.hidden_size}")
    logging.info(f"  - LSTM layers: {total_lstm_layers}")
    logging.info(f"  - Sequence length: {args.seq_len}")
    logging.info(f"  - Dropout: {args.dropout}")
    logging.info(f"  - Use attention: {args.use_attention}")
    logging.info(f"  - Bidirectional: {args.bidirectional}")
    
    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.95)
    )
    
    # 5. Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # 6. Train
    trainer = LSTMTrainer(model, optimizer, device, args.checkpoint_dir)
    
    logging.info("Starting LSTM baseline training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(tokens, labels, epoch)
        scheduler.step()
        
        # Track best model
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                model.state_dict(), 
                os.path.join(args.checkpoint_dir, "best_lstm_baseline.pt")
            )
        
        logging.info(
            f"Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.2e} "
            f"- Loss: {metrics['loss']:.4f} - Acc: {metrics['accuracy']:.3f} "
            f"- Best Loss: {best_loss:.4f}"
        )
    
    logging.info("LSTM baseline training completed successfully!")
    logging.info(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Baseline Model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Close")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    
    # VQ-VAE parameters (must match the trained VQ-VAE)
    parser.add_argument("--vq_hiddens", type=int, default=64)
    parser.add_argument("--vq_res_layers", type=int, default=2)
    parser.add_argument("--vq_res_hiddens", type=int, default=32)
    parser.add_argument("--vq_emb_dim", type=int, default=64)
    
    # Model parameters (match HRM configuration)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--l_layers", type=int, default=2)
    parser.add_argument("--h_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # LSTM-specific parameters
    parser.add_argument("--use_attention", action="store_true",
                       help="Add attention mechanism to LSTM")
    parser.add_argument("--bidirectional", action="store_true",
                       help="Use bidirectional LSTM")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_lstm_baseline")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)

