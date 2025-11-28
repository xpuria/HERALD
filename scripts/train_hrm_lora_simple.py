"""
Simplified LoRA training for HRM - Direct approach without ACTLossHead complexity
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.hrm import HierarchicalReasoningModel_ACTV1
from src.models.lora import add_lora_to_linear, count_lora_parameters
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging
from tqdm import tqdm
import logging


def simple_train_epoch(model, tokens, labels, optimizer, device, epoch, seq_len=16):
    """Simplified training loop with direct loss computation"""
    model.train()
    
    batch_size, total_steps = tokens.shape
    num_segments = total_steps // seq_len
    
    carry = None
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    valid_batches = 0
    
    progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
    
    for i in progress_bar:
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        input_tokens = tokens[:, start_idx:end_idx].to(device)
        target_labels = labels[:, start_idx:end_idx].to(device)
        
        # Create batch
        batch = {
            "inputs": input_tokens,
            "labels": target_labels
        }
        
        # Initialize/reset carry
        if carry is None or i % 20 == 0:
            carry = model.initial_carry(batch)
        
        try:
            optimizer.zero_grad()
            
            # Forward pass
            new_carry, outputs = model(carry=carry, batch=batch)
            carry = new_carry
            
            # Simple cross-entropy loss (averaged over batch and sequence)
            logits = outputs["logits"]  # (batch, seq, vocab)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = target_labels.reshape(-1)
            
            # Compute loss with proper reduction
            loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
            
            # Skip if NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected, skipping batch {i}")
                carry = None
                continue
            
            # Backward
            loss.backward()
            
            # Gradient clipping - use aggressive clipping to handle large gradients
            # Note: Initial gradients are expected to be large (50-70) for untrained models
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Only skip if gradients are truly extreme (>500, indicating numerical issues)
            if grad_norm > 500.0:
                logging.warning(f"Extreme gradient: {grad_norm:.2f}, skipping")
                carry = None
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
            carry = None
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
    setup_logging(os.path.join(args.checkpoint_dir, "train_simple.log"))
    
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
    
    # Create HRM
    logging.info("Creating HRM model...")
    config_dict = {
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'vocab_size': 512,
        'H_cycles': 2,
        'L_cycles': 4,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 128,
        'expansion': 4.0,
        'num_heads': 4,
        'halt_max_steps': 1,
        'halt_exploration_prob': 0.0,
        'forward_dtype': 'float32'
    }
    
    model = HierarchicalReasoningModel_ACTV1(config_dict).to(device)
    
    # Add LoRA
    logging.info("Adding LoRA adapters...")
    model = add_lora_to_linear(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        freeze_base=True,
        target_modules=['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )
    
    # Count parameters
    param_stats = count_lora_parameters(model)
    logging.info(f"Total parameters: {param_stats['total_params']:,}")
    logging.info(f"Trainable (LoRA) parameters: {param_stats['lora_params']:,}")
    logging.info(f"Trainable ratio: {param_stats['trainable_ratio']:.2%}")
    
    if param_stats['lora_params'] == 0:
        logging.error("No LoRA parameters added! Check module names.")
        return
    
    # Optimizer (only LoRA parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable tensors: {len(trainable_params)}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # Train
    logging.info("Starting training...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = simple_train_epoch(
            model, tokens, labels, optimizer, device, epoch, args.seq_len
        )
        scheduler.step()
        
        # Save checkpoint
        if metrics["loss"] < best_loss and metrics["valid_batches"] > 0:
            best_loss = metrics["loss"]
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }, os.path.join(args.checkpoint_dir, "best_lora.pt"))
            
            # Save only LoRA weights
            lora_params = {
                name: param for name, param in model.named_parameters()
                if 'lora' in name.lower()
            }
            torch.save(lora_params, os.path.join(args.checkpoint_dir, "best_lora_only.pt"))
        
        logging.info(
            f"Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.2e} - "
            f"Loss: {metrics['loss']:.4f} - Acc: {metrics['accuracy']:.3f} - "
            f"Valid: {metrics['valid_batches']} - Best: {best_loss:.4f}"
        )
    
    logging.info(f"Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Close")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_hrm_lora")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)

