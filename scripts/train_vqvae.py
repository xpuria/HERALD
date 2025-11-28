"""
train vq-vae tokenizer on financial time series
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import sys
sys.path.append('.')

from src.data.dataset import FinancialDataset
from src.models.vq_vae import VQVAE
from src.utils import set_seed, setup_logging


def train(args):
    set_seed(args.seed)
    setup_logging(os.path.join(args.checkpoint_dir, "train.log"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = FinancialDataset(
        csv_path=args.data_path,
        seq_len=args.seq_len,
        mode='train',
        use_zigzag=False,
        target_col=args.target_col
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = VQVAE(
        num_hiddens=args.num_hiddens,
        num_residual_layers=args.num_residual_layers,
        num_residual_hiddens=args.num_residual_hiddens,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x, _ in progress_bar:
            x = x.to(device)
            
            optimizer.zero_grad()
            vq_loss, x_recon, perplexity, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), recon=recon_loss.item(), vq=vq_loss.item(), ppl=perplexity.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"vqvae_epoch_{epoch+1}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="Close")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    parser.add_argument("--num_hiddens", type=int, default=64)
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--num_residual_hiddens", type=int, default=32)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    
    args = parser.parse_args()
    train(args)
