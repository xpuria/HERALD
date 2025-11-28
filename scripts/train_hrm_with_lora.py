"""
Train HRM with LoRA (Low-Rank Adaptation)

This script trains the HRM model with LoRA adapters, which allows efficient
fine-tuning with a fraction of the parameters.
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.hrm import HierarchicalReasoningModel_ACTV1, ACTLossHead
from src.models.lora import add_lora_to_linear, count_lora_parameters
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging
from tqdm import tqdm
import logging


def create_hrm_with_lora(config_dict, lora_config):
    """
    Create HRM model with LoRA adapters.
    
    Args:
        config_dict: HRM configuration
        lora_config: Dictionary with LoRA settings
            - rank: LoRA rank
            - alpha: LoRA alpha scaling
            - dropout: LoRA dropout
            - target_modules: List of module names to apply LoRA to
            
    Returns:
        HRM model with LoRA adapters
    """
    # Create base HRM model
    model = HierarchicalReasoningModel_ACTV1(config_dict)
    
    # Add LoRA to specified modules
    model = add_lora_to_linear(
        model,
        rank=lora_config['rank'],
        alpha=lora_config['alpha'],
        dropout=lora_config['dropout'],
        freeze_base=True,  # Freeze base weights
        target_modules=lora_config['target_modules']
    )
    
    return model


class LoRAHRMTrainer:
    """Trainer for HRM with LoRA"""
    
    def __init__(self, model, optimizer, device, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Wrap model with loss head
        self.loss_model = ACTLossHead(self.model)
        
    def train_epoch(self, tokens, labels, epoch):
        """Train one epoch"""
        self.loss_model.train()
        
        batch_size, total_steps = tokens.shape
        seq_len = self.model.config.seq_len
        num_segments = total_steps // seq_len
        
        # Initialize carry
        carry = None
        
        epoch_metrics = {
            "lm_loss": 0.0,
            "accuracy": 0.0,
            "exact_accuracy": 0.0,
            "valid_batches": 0
        }
        
        progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
        
        for i in progress_bar:
            # Get segment
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(self.device)
            target_labels = labels[:, start_idx:end_idx].to(self.device)
            
            # Create batch
            batch = {
                "inputs": input_tokens,
                "labels": target_labels
            }
            
            # Initialize carry if needed
            if carry is None:
                carry = self.loss_model.initial_carry(batch)
            
            # Reset carry periodically for stability
            if i % 20 == 0:
                carry = self.loss_model.initial_carry(batch)
            
            try:
                self.optimizer.zero_grad()
                
                # Forward pass
                new_carry, loss, metrics, outputs, all_halted = self.loss_model(
                    return_keys=[],
                    carry=carry,
                    batch=batch
                )
                
                carry = new_carry
                
                # Check for NaN/Inf only (remove overly strict loss threshold)
                # Initial losses can be high (160+) for untrained models - this is normal!
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"NaN/Inf loss detected: {loss:.2e}, resetting carry")
                    carry = None
                    continue
                
                # Only reject extremely high losses (10x typical range)
                if loss > 1000:
                    logging.warning(f"Extremely high loss: {loss:.2e}, resetting carry")
                    carry = None
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping (aggressive for LoRA training)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=0.5
                )
                
                if grad_norm > 5.0:
                    logging.warning(f"Large gradient norm: {grad_norm:.2f}, skipping update")
                    carry = None
                    continue
                
                self.optimizer.step()
                
                # Update metrics
                for key in ["lm_loss", "accuracy", "exact_accuracy"]:
                    if key in metrics:
                        value = metrics[key].item() if torch.is_tensor(metrics[key]) else metrics[key]
                        epoch_metrics[key] += value
                epoch_metrics["valid_batches"] += 1
                
                # Normalize accuracy by count
                count = metrics.get("count", batch_size)
                count = count.item() if torch.is_tensor(count) else count
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{(metrics['accuracy'] / count).item() if count > 0 else 0:.3f}",
                    "grad_norm": f"{grad_norm:.2f}"
                })
                
            except Exception as e:
                logging.error(f"Error in batch {i}: {e}")
                carry = None
                continue
        
        # Compute averages
        if epoch_metrics["valid_batches"] > 0:
            for key in epoch_metrics:
                if key != "valid_batches":
                    epoch_metrics[key] /= epoch_metrics["valid_batches"]
        
        logging.info(f"Epoch {epoch} completed: {epoch_metrics}")
        
        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save full model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics': epoch_metrics
        }, os.path.join(self.checkpoint_dir, f"hrm_lora_epoch_{epoch}.pt"))
        
        # Also save just LoRA weights (much smaller!)
        lora_params = {
            name: param for name, param in self.model.named_parameters()
            if 'lora' in name.lower()
        }
        torch.save({
            'lora_weights': lora_params,
            'epoch': epoch,
            'metrics': epoch_metrics
        }, os.path.join(self.checkpoint_dir, f"lora_only_epoch_{epoch}.pt"))
        
        return epoch_metrics


def main(args):
    set_seed(args.seed)
    setup_logging(os.path.join(args.checkpoint_dir, "train_hrm_lora.log"))
    
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
    
    # 3. Initialize HRM with LoRA
    logging.info("Initializing HRM with LoRA...")
    
    config_dict = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': args.vocab_size,
        'H_cycles': args.h_cycles,
        'L_cycles': args.l_cycles,
        'H_layers': args.h_layers,
        'L_layers': args.l_layers,
        'hidden_size': args.hidden_size,
        'expansion': args.expansion,
        'num_heads': args.num_heads,
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'forward_dtype': 'float32'
    }
    
    lora_config = {
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'dropout': args.lora_dropout,
        'target_modules': args.lora_target_modules
    }
    
    model = create_hrm_with_lora(config_dict, lora_config).to(device)
    
    # Count parameters
    param_stats = count_lora_parameters(model)
    logging.info(f"Parameter Statistics:")
    logging.info(f"  Total parameters: {param_stats['total_params']:,}")
    logging.info(f"  Trainable parameters: {param_stats['trainable_params']:,}")
    logging.info(f"  LoRA parameters: {param_stats['lora_params']:,}")
    logging.info(f"  Trainable ratio: {param_stats['trainable_ratio']:.2%}")
    logging.info(f"  Memory saving: {1 - param_stats['trainable_ratio']:.2%}")
    
    logging.info(f"\nLoRA Configuration:")
    logging.info(f"  Rank: {args.lora_rank}")
    logging.info(f"  Alpha: {args.lora_alpha}")
    logging.info(f"  Dropout: {args.lora_dropout}")
    logging.info(f"  Target modules: {args.lora_target_modules}")
    
    # 4. Optimizer (only for LoRA parameters)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    trainer = LoRAHRMTrainer(model, optimizer, device, args.checkpoint_dir)
    
    logging.info("Starting HRM training with LoRA...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(tokens, labels, epoch)
        scheduler.step()
        
        # Track best model
        if metrics["lm_loss"] < best_loss:
            best_loss = metrics["lm_loss"]
            torch.save(
                model.state_dict(),
                os.path.join(args.checkpoint_dir, "best_hrm_lora.pt")
            )
            # Also save best LoRA-only weights
            lora_params = {
                name: param for name, param in model.named_parameters()
                if 'lora' in name.lower()
            }
            torch.save(
                lora_params,
                os.path.join(args.checkpoint_dir, "best_lora_only.pt")
            )
        
        logging.info(
            f"Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.2e} "
            f"- Loss: {metrics['lm_loss']:.4f} - Acc: {metrics['accuracy']:.3f} "
            f"- Best Loss: {best_loss:.4f}"
        )
    
    logging.info("HRM LoRA training completed successfully!")
    logging.info(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRM with LoRA")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Close")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    
    # VQ-VAE parameters
    parser.add_argument("--vq_hiddens", type=int, default=64)
    parser.add_argument("--vq_res_layers", type=int, default=2)
    parser.add_argument("--vq_res_hiddens", type=int, default=32)
    parser.add_argument("--vq_emb_dim", type=int, default=64)
    
    # HRM parameters (match existing configuration)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--h_cycles", type=int, default=2)
    parser.add_argument("--l_cycles", type=int, default=4)
    parser.add_argument("--h_layers", type=int, default=2)
    parser.add_argument("--l_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--expansion", type=float, default=4.0)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # LoRA-specific parameters
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha scaling (typically 2*rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout for regularization")
    parser.add_argument("--lora_target_modules", nargs='+',
                       default=['qkv_proj', 'o_proj'],
                       help="Which modules to apply LoRA to")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=20,
                       help="More epochs for LoRA (needs more iterations)")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate (can be higher for LoRA)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_hrm_lora")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)

