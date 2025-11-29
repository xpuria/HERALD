"""
lstm baseline model for financial time series prediction

this lstm implementation uses the same hyperparameters as the hrm model
to provide a fair comparison baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """configuration for lstm baseline matching hrm hyperparameters."""
    vocab_size: int         # input vocabulary size (from vq-vae)
    hidden_size: int        # hidden dimension (matches hrm hidden_size)
    num_layers: int         # number of lstm layers
    dropout: float = 0.1    # dropout probability
    bidirectional: bool = False  # whether to use bidirectional lstm
    seq_len: int = 16       # sequence length


class LSTMBaseline(nn.Module):
    """
    lstm baseline model for comparison with hrm.
    
    this model uses the same capacity as hrm but with standard lstm architecture:
    - same embedding dimension (hidden_size)
    - comparable number of layers
    - same output heads (binary classification)
    
    architecture:
        token embedding → lstm layers → classification head
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # token embedding (same as hrm)
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        
        # initialize with scaled normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # positional embedding
        self.position_embedding = nn.Embedding(
            config.seq_len * 2,  # allow longer sequences
            config.hidden_size
        )
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # lstm layers with dropout
        # note: dropout is applied between lstm layers, not on the output
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional
        )
        
        # hidden state multiplier for bidirectional lstm
        self.direction_multiplier = 2 if config.bidirectional else 1
        lstm_output_size = config.hidden_size * self.direction_multiplier
        
        # layer normalization for stability
        self.output_norm = nn.LayerNorm(lstm_output_size)
        
        # output head for binary classification (up/down)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 2)  # binary classification
        )
        
        # halt prediction head (for consistency with hrm interface)
        self.halt_head = nn.Linear(lstm_output_size, 1)
        
        # initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """initialize lstm and linear layers with conservative values."""
        # initialize lstm weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # hidden-to-hidden weights (use orthogonal for stability)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # initialize biases to zero, except forget gate bias
                nn.init.zeros_(param)
                # set forget gate bias to 1 (standard lstm practice)
                n = param.size(0)
                forget_gate_start = n // 4
                forget_gate_end = n // 2
                param.data[forget_gate_start:forget_gate_end].fill_(1.0)
        
        # initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # initialize halt head
        nn.init.normal_(self.halt_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.halt_head.bias, -5.0)  # start with low halt probability
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> dict:
        """
        forward pass through lstm.
        
        args:
            input_ids: (batch_size, seq_len) token indices
            hidden_state: optional tuple of (h_0, c_0) lstm states
                         each is (num_layers * num_directions, batch_size, hidden_size)
        
        returns:
            dictionary containing:
                - logits: (batch_size, seq_len, 2) classification logits
                - halt_probs: (batch_size, 1) halt probabilities
                - hidden_state: new lstm hidden state for next iteration
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # token embeddings
        token_embeds = self.token_embedding(input_ids)  # (b, l, d)
        
        # positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)  # (b, l, d)
        
        # combine embeddings
        x = token_embeds + pos_embeds  # (b, l, d)
        
        # lstm forward pass
        if hidden_state is not None:
            lstm_out, new_hidden = self.lstm(x, hidden_state)
        else:
            lstm_out, new_hidden = self.lstm(x)
        
        # lstm_out: (b, l, d * num_directions)
        # new_hidden: tuple of (h_n, c_n)
        
        # normalize lstm output
        lstm_out = self.output_norm(lstm_out)
        
        # classification logits for each timestep
        logits = self.classifier(lstm_out)  # (b, l, 2)
        
        # halt prediction (use last timestep or mean)
        halt_logits = self.halt_head(lstm_out.mean(dim=1))  # (b, 1)
        halt_probs = torch.sigmoid(halt_logits)
        
        return {
            'logits': logits,
            'halt_probs': halt_probs,
            'hidden_state': new_hidden
        }
    
    def count_parameters(self) -> int:
        """count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
