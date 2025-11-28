"""
dataset classes for financial time series

- FinancialDataset: for vq-vae training (raw preprocessed data)
- HRMDataset: for reasoning model (tokenized via vq-vae)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict

from .preprocessing import preprocess_pipeline
from .zigzag import zigzag


class FinancialDataset(Dataset):
    """dataset for vq-vae training with preprocessed windows"""
    
    def __init__(self, 
                 csv_path: str, 
                 seq_len: int = 64, 
                 mode: str = 'train',
                 preprocessing_window: int = 30,
                 zigzag_deviation: float = 0.01,
                 use_zigzag: bool = True,
                 target_col: str = 'Close'):
        self.seq_len = seq_len
        self.mode = mode
        
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' not found. available: {list(df.columns)}")
        prices = df[target_col].values
        
        self.features = preprocess_pipeline(prices, window=preprocessing_window)
        
        if use_zigzag:
            _, self.labels = zigzag(prices, deviation_pct=zigzag_deviation)
        else:
            self.labels = np.zeros(len(prices))
        
        self.n_samples = len(self.features) - seq_len
        
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx : idx + self.seq_len]
        
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor


class HRMDataset(Dataset):
    """dataset for hrm training with vq-vae tokenization (64 -> 16 tokens)"""
    
    def __init__(self, 
                 csv_path: str, 
                 vqvae_model: Any,
                 seq_len: int = 64, 
                 preprocessing_window: int = 30,
                 zigzag_deviation: float = 0.01,
                 device: str = 'cpu',
                 target_col: str = 'Close'):
        self.seq_len = seq_len
        self.device = device
        
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"target column '{target_col}' not found.")
        prices = df[target_col].values
        features = preprocess_pipeline(prices, window=preprocessing_window)
        _, labels = zigzag(prices, deviation_pct=zigzag_deviation)
        
        self.features = features
        self.labels = labels
        self.vqvae = vqvae_model
        self.vqvae.eval()
        
        self.n_samples = len(self.features) - seq_len

    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx : idx + self.seq_len]
        
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1).to(self.device)
        
        with torch.no_grad():
            _, _, _, encoding_indices = self.vqvae(x_tensor)
        
        tokens = encoding_indices.view(-1).cpu()
        
        # downsample labels to match tokens (64 -> 16)
        downsample_factor = len(x) // len(tokens)
        y_downsampled = y[downsample_factor-1::downsample_factor]
        if len(y_downsampled) != len(tokens):
            y_downsampled = y_downsampled[:len(tokens)]
        
        return {
            "inputs": tokens.long(),
            "labels": torch.tensor(y_downsampled, dtype=torch.long),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long)
        }

    def get_all_tokens_and_labels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize entire dataset at once for efficiency"""
        all_tokens = []
        all_labels = []
        
        step = 64
        total_len = len(self.features)
        batch_size = 32
        
        print("tokenizing entire dataset...")
        
        for i in range(0, total_len - 64, step * batch_size):
            batch_x = []
            batch_y = []
            
            for b in range(batch_size):
                idx = i + b * step
                if idx + 64 > total_len:
                    break
                    
                x = self.features[idx : idx + 64]
                y = self.labels[idx : idx + 64]
                
                batch_x.append(torch.tensor(x, dtype=torch.float32).view(1, -1))
                
                y_down = y[3::4]
                if len(y_down) > 16:
                    y_down = y_down[:16]
                batch_y.append(torch.tensor(y_down, dtype=torch.long))
            
            if not batch_x:
                break
            
            x_tensor = torch.stack(batch_x).to(self.device)
            
            with torch.no_grad():
                _, _, _, encoding_indices = self.vqvae(x_tensor)
            
            tokens = encoding_indices.view(len(batch_x), -1).cpu()
            labels = torch.stack(batch_y)
            
            all_tokens.append(tokens)
            all_labels.append(labels)
        
        full_tokens = torch.cat(all_tokens, dim=0).view(-1)
        full_labels = torch.cat(all_labels, dim=0).view(-1)
        
        return full_tokens, full_labels
