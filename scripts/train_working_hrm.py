"""
train hierarchical reasoning model on vq-vae tokens
"""

import argparse
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('.')

from src.models.working_hrm import WorkingHierarchicalReasoningModel, WorkingHRMState
from src.models.hrm import HierarchicalReasoningModel_ACTV1Config
from src.models.vq_vae import VQVAE
from src.data.dataset import HRMDataset
from src.utils import set_seed, setup_logging
from tqdm import tqdm
import logging


class WorkingHRMTrainer:
    def __init__(self, model, optimizer, device, checkpoint_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
    def compute_loss(self, outputs, labels):
        logits = outputs["logits"]
        halt_probs = outputs["halt_probs"]
        
        logits_flat = logits.reshape(-1, 2)
        labels_flat = labels.reshape(-1)
        
        classification_loss = F.cross_entropy(
            logits_flat, labels_flat, label_smoothing=0.1, reduction='mean'
        )
        halt_reg_loss = F.mse_loss(halt_probs, torch.full_like(halt_probs, 0.5))
        total_loss = classification_loss + 0.01 * halt_reg_loss
        
        with torch.no_grad():
            predictions = torch.argmax(logits_flat, dim=1)
            accuracy = (predictions == labels_flat).float().mean()
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "halt_reg_loss": halt_reg_loss,
            "accuracy": accuracy
        }
    
    def train_epoch(self, tokens, labels, epoch):
        self.model.train()
        
        batch_size, total_steps = tokens.shape
        seq_len = self.model.config.seq_len
        num_segments = total_steps // seq_len
        
        state = None
        epoch_metrics = {
            "total_loss": 0.0, "classification_loss": 0.0,
            "halt_reg_loss": 0.0, "accuracy": 0.0, "valid_batches": 0
        }
        
        progress_bar = tqdm(range(num_segments), desc=f"Epoch {epoch}")
        
        for i in progress_bar:
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            
            input_tokens = tokens[:, start_idx:end_idx].to(self.device)
            target_labels = labels[:, start_idx:end_idx].to(self.device)
            
            if i % 20 == 0:
                state = None
            
            try:
                self.optimizer.zero_grad()
                outputs = self.model(input_tokens, state)
                state = outputs["state"]
                
                loss_dict = self.compute_loss(outputs, target_labels)
                total_loss = loss_dict["total_loss"]
                
                if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 100:
                    logging.warning(f"unstable loss: {total_loss:.2e}, resetting state")
                    state = None
                    continue
                
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                if grad_norm > 5.0:
                    logging.warning(f"large gradient norm: {grad_norm:.2f}, skipping update")
                    state = None
                    continue
                
                self.optimizer.step()
                
                for key in epoch_metrics:
                    if key in loss_dict:
                        epoch_metrics[key] += loss_dict[key].item()
                epoch_metrics["valid_batches"] += 1
                
                progress_bar.set_postfix({
                    "loss": f"{total_loss.item():.4f}",
                    "acc": f"{loss_dict['accuracy'].item():.3f}",
                    "grad_norm": f"{grad_norm:.2f}",
                    "state_norm": f"{state.z_H.norm().item():.2f}" if state else "0.00"
                })
                
            except Exception as e:
                logging.error(f"error in batch {i}: {e}")
                state = None
                continue
        
        if epoch_metrics["valid_batches"] > 0:
            for key in epoch_metrics:
                if key != "valid_batches":
                    epoch_metrics[key] /= epoch_metrics["valid_batches"]
        
        logging.info(f"Epoch {epoch} completed: {epoch_metrics}")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics': epoch_metrics
        }, os.path.join(self.checkpoint_dir, f"working_hrm_epoch_{epoch}.pt"))
        
        return epoch_metrics


def main(args):
    set_seed(args.seed)
    setup_logging(os.path.join(args.checkpoint_dir, "train_working_hrm.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")
    
    logging.info("loading vq-vae...")
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
        logging.info(f"loaded vq-vae from {args.vqvae_checkpoint}")
    
    logging.info("preparing data...")
    dataset = HRMDataset(
        csv_path=args.data_path,
        vqvae_model=vqvae,
        seq_len=64,
        device=str(device),
        target_col=args.target_col
    )
    
    tokens, labels = dataset.get_all_tokens_and_labels()
    logging.info(f"total tokens: {tokens.shape[0]}")
    
    batch_size = args.batch_size
    stream_len = tokens.shape[0] // batch_size
    tokens = tokens[:batch_size * stream_len].reshape(batch_size, stream_len)
    labels = labels[:batch_size * stream_len].reshape(batch_size, stream_len)
    
    logging.info("initializing hrm...")
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        H_layers=args.h_layers,
        L_layers=args.l_layers,
        hidden_size=args.hidden_size,
        expansion=args.expansion,
        num_heads=args.num_heads,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32"
    )
    
    model = WorkingHierarchicalReasoningModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        eps=1e-8, betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    trainer = WorkingHRMTrainer(model, optimizer, device, args.checkpoint_dir)
    
    logging.info("starting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(tokens, labels, epoch)
        scheduler.step()
        
        if metrics["total_loss"] < best_loss:
            best_loss = metrics["total_loss"]
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_working_hrm.pt"))
        
        logging.info(f"Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.2e} - Best Loss: {best_loss:.4f}")
    
    logging.info("training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Close")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    
    parser.add_argument("--vq_hiddens", type=int, default=64)
    parser.add_argument("--vq_res_layers", type=int, default=2)
    parser.add_argument("--vq_res_hiddens", type=int, default=32)
    parser.add_argument("--vq_emb_dim", type=int, default=64)
    
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
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_working_hrm")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)
