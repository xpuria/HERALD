"""
multifeature patch-hrm: same architecture as patch_hrm but accepts (B, L, C) input.
each non-overlapping length-`patch_size` window is flattened across feature
channels then linearly projected to hidden_size.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.working_hrm import StabilizedLayer, WorkingHRMState


@dataclass
class MultiFeatureHRMConfig:
    raw_seq_len: int = 64
    patch_size: int = 4
    num_features: int = 6
    hidden_size: int = 128
    num_heads: int = 4
    expansion: float = 4.0
    H_cycles: int = 2
    L_cycles: int = 4
    H_layers: int = 2
    L_layers: int = 2
    num_classes: int = 2


class MultiFeatureHRM(nn.Module):
    def __init__(self, cfg: MultiFeatureHRMConfig):
        super().__init__()
        assert cfg.raw_seq_len % cfg.patch_size == 0
        self.cfg = cfg
        self.num_patches = cfg.raw_seq_len // cfg.patch_size

        self.patch_embed = nn.Linear(cfg.patch_size * cfg.num_features, cfg.hidden_size)
        self.position_embedding = nn.Embedding(self.num_patches, cfg.hidden_size)

        self.L_layers = nn.ModuleList([
            StabilizedLayer(cfg.hidden_size, cfg.num_heads, cfg.expansion)
            for _ in range(cfg.L_layers)
        ])
        self.H_layers = nn.ModuleList([
            StabilizedLayer(cfg.hidden_size, cfg.num_heads, cfg.expansion)
            for _ in range(cfg.H_layers)
        ])
        self.L_to_H_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.H_to_L_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.state_mixer = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.num_classes, bias=True)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int, device: torch.device) -> WorkingHRMState:
        return WorkingHRMState(
            z_H=torch.zeros(batch_size, self.num_patches, self.cfg.hidden_size, device=device),
            z_L=torch.zeros(batch_size, self.num_patches, self.cfg.hidden_size, device=device),
        )

    def forward(self, raw: torch.Tensor, state: Optional[WorkingHRMState] = None) -> Dict[str, torch.Tensor]:
        # raw: (B, raw_seq_len, num_features)
        B, L, C = raw.shape
        assert L == self.cfg.raw_seq_len and C == self.cfg.num_features
        device = raw.device

        if state is None or state.z_L.shape[0] != B:
            state = self.initial_state(B, device)

        patches = raw.view(B, self.num_patches, self.cfg.patch_size * C)
        tok = self.patch_embed(patches)
        pos = self.position_embedding(torch.arange(self.num_patches, device=device)).unsqueeze(0)
        x = tok + pos

        z_L = x + state.z_L * 0.1
        z_H = state.z_H * 0.1

        for cycle in range(self.cfg.L_cycles):
            z_in = z_L
            for layer in self.L_layers:
                z_in = layer(z_in, causal_mask=False)
            z_L = z_L + z_in * 0.5

            every = max(1, self.cfg.L_cycles // self.cfg.H_cycles)
            if (cycle + 1) % every == 0:
                z_in = z_H + self.L_to_H_proj(z_L)
                for layer in self.H_layers:
                    z_in = layer(z_in, causal_mask=False)
                z_H = z_H + z_in * 0.5
                z_L = z_L + self.H_to_L_proj(z_H) * 0.1

        mixed = self.state_mixer(torch.cat([z_H, z_L], dim=-1))
        logits = self.lm_head(self.output_norm(mixed))
        new_state = WorkingHRMState(z_H=z_H.detach() * 0.9, z_L=z_L.detach() * 0.9)
        return {"logits": logits, "state": new_state}
