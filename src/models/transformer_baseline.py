"""
transformer baseline model for financial time series prediction

this transformer implementation uses the same hyperparameters as the hrm model
to provide a fair comparison baseline alongside lstm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class TransformerConfig:
    """configuration for transformer baseline matching hrm hyperparameters."""
    vocab_size: int         # input vocabulary size (from vq-vae)
    hidden_size: int        # hidden dimension (matches hrm hidden_size)
    num_layers: int         # number of transformer layers
    num_heads: int          # number of attention heads
    ff_dim: int            # feedforward dimension
    dropout: float = 0.1    # dropout probability
    seq_len: int = 16       # sequence length
    use_causal_mask: bool = True  # use causal masking for autoregressive prediction


class PositionalEncoding(nn.Module):
    """sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: (batch_size, seq_len, d_model)
        returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    transformer baseline model for comparison with hrm and lstm.
    
    this model uses the same capacity as hrm but with standard transformer architecture:
    - same embedding dimension (hidden_size)
    - comparable number of layers
    - same output heads (binary classification)
    
    architecture:
        token embedding → positional encoding → transformer layers → classification head
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # token embedding (same as hrm/lstm)
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        
        # initialize with scaled normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(
            config.hidden_size,
            max_len=config.seq_len * 2,
            dropout=config.dropout
        )
        
        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # pre-ln for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_size)
        )
        
        # output head for binary classification (up/down)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 2)  # binary classification
        )
        
        # halt prediction head (for consistency with hrm interface)
        self.halt_head = nn.Linear(config.hidden_size, 1)
        
        # initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """initialize weights with conservative values."""
        # classifier initialization
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # halt head initialization
        nn.init.normal_(self.halt_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.halt_head.bias, -5.0)  # start with low halt probability
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        forward pass through transformer.
        
        args:
            input_ids: (batch_size, seq_len) token indices
            attention_mask: optional (seq_len, seq_len) or (batch_size, seq_len, seq_len)
                          if none and use_causal_mask=true, creates causal mask
        
        returns:
            dictionary containing:
                - logits: (batch_size, seq_len, 2) classification logits
                - halt_probs: (batch_size, 1) halt probabilities
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # token embeddings
        x = self.token_embedding(input_ids)  # (b, l, d)
        
        # add positional encoding
        x = self.pos_encoder(x)  # (b, l, d)
        
        # create causal mask if needed
        if attention_mask is None and self.config.use_causal_mask:
            # create causal mask: upper triangular matrix with -inf
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
        
        # transformer forward pass
        # pytorch transformerencoder expects mask shape (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
        transformer_out = self.transformer(x, mask=attention_mask)  # (b, l, d)
        
        # classification logits for each timestep
        logits = self.classifier(transformer_out)  # (b, l, 2)
        
        # halt prediction (use mean pooling over sequence)
        halt_logits = self.halt_head(transformer_out.mean(dim=1))  # (b, 1)
        halt_probs = torch.sigmoid(halt_logits)
        
        return {
            'logits': logits,
            'halt_probs': halt_probs
        }
    
    def count_parameters(self) -> int:
        """count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
