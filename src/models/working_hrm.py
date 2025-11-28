"""
fin-hrm: financial hierarchical reasoning model

adapts hrm architecture for financial time series prediction using
fast (l-module) and slow (h-module) processing modules.

based on: https://github.com/sapientinc/HRM/tree/main/models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math

from src.models.layers import CastedEmbedding
from src.models.hrm import HierarchicalReasoningModel_ACTV1Config


@dataclass
class WorkingHRMState:
    """model's memory between chunks"""
    z_H: torch.Tensor  # slow module state
    z_L: torch.Tensor  # fast module state


class StabilizedLayer(nn.Module):
    """transformer layer with pre-norm and dropout for stability"""
    
    def __init__(self, hidden_size: int, num_heads: int, expansion: float = 4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        intermediate_size = int(hidden_size * expansion)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        for module in [self.qkv_proj, self.o_proj, self.gate_proj, self.up_proj, self.down_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)

        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal_mask:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L, D)
        attn_output = self.o_proj(attn_output)

        x = residual + self.dropout(attn_output)

        residual = x
        x = self.post_attention_layernorm(x)

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_output = self.down_proj(gate * up)

        x = residual + self.dropout(mlp_output)

        return x


class WorkingHierarchicalReasoningModel(nn.Module):
    """
    hierarchical model with fast (l) and slow (h) modules.
    l-module runs 4 cycles for immediate patterns.
    h-module runs 2 cycles for longer-term context.
    they exchange information through projections.
    """
    
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.seq_len * 2, config.hidden_size)

        self.L_layers = nn.ModuleList([
            StabilizedLayer(config.hidden_size, config.num_heads, config.expansion)
            for _ in range(config.L_layers)
        ])

        self.H_layers = nn.ModuleList([
            StabilizedLayer(config.hidden_size, config.num_heads, config.expansion)
            for _ in range(config.H_layers)
        ])

        self.L_to_H_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.H_to_L_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.state_mixer = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, 2, bias=True)
        self.halt_head = nn.Linear(config.hidden_size, 1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, seq_len: int, device: torch.device) -> WorkingHRMState:
        return WorkingHRMState(
            z_H=torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device),
            z_L=torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
        )

    def forward(self, input_ids: torch.Tensor, state: Optional[WorkingHRMState] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if state is None:
            state = self.initial_state(batch_size, seq_len, device)

        if state.z_L.shape[1] != seq_len:
            state = self.initial_state(batch_size, seq_len, device)

        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        # mix in previous state
        z_L = x + state.z_L * 0.1
        z_H = state.z_H * 0.1

        # hierarchical reasoning cycles
        for cycle in range(self.config.L_cycles):
            z_L_input = z_L
            for layer in self.L_layers:
                z_L_input = layer(z_L_input, causal_mask=True)

            z_L = z_L + z_L_input * 0.5

            # h-module processes every few l cycles
            if (cycle + 1) % max(1, self.config.L_cycles // self.config.H_cycles) == 0:
                h_input = self.L_to_H_proj(z_L)
                z_H_input = z_H + h_input

                for layer in self.H_layers:
                    z_H_input = layer(z_H_input, causal_mask=True)

                z_H = z_H + z_H_input * 0.5

                # feedback from h to l
                l_feedback = self.H_to_L_proj(z_H)
                z_L = z_L + l_feedback * 0.1

        combined_state = torch.cat([z_H, z_L], dim=-1)
        final_state = self.state_mixer(combined_state)

        output = self.output_norm(final_state)
        logits = self.lm_head(output)

        halt_logits = self.halt_head(output.mean(dim=1))
        halt_probs = torch.sigmoid(halt_logits)

        new_state = WorkingHRMState(
            z_H=z_H.detach() * 0.9,
            z_L=z_L.detach() * 0.9
        )

        return {
            "logits": logits,
            "halt_probs": halt_probs,
            "state": new_state
        }
