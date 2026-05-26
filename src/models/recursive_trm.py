"""
recursive-trm: a single transformer block iterated K times (tiny recursive
models, samsung sail 2025). replaces the H+L two-stack hierarchy with one
shared block, trading parameter count for compute. inputs are continuous
patches (same as patch-hrm), so this branch isolates the architectural
change from the input-representation change.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.working_hrm import StabilizedLayer, WorkingHRMState


@dataclass
class RecursiveTRMConfig:
    raw_seq_len: int = 64
    patch_size: int = 4
    hidden_size: int = 128
    num_heads: int = 4
    expansion: float = 4.0
    num_layers: int = 2          # layers inside the shared block
    num_cycles: int = 8          # number of times the block is reapplied
    num_classes: int = 2


class RecursiveTRM(nn.Module):
    def __init__(self, cfg: RecursiveTRMConfig):
        super().__init__()
        assert cfg.raw_seq_len % cfg.patch_size == 0
        self.cfg = cfg
        self.num_patches = cfg.raw_seq_len // cfg.patch_size

        self.patch_embed = nn.Linear(cfg.patch_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(self.num_patches, cfg.hidden_size)

        # one shared block applied num_cycles times
        self.block = nn.ModuleList([
            StabilizedLayer(cfg.hidden_size, cfg.num_heads, cfg.expansion)
            for _ in range(cfg.num_layers)
        ])
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
        # state is unused for this model (no recurrence across windows); kept for
        # interface compatibility with the training/eval harness.
        return WorkingHRMState(
            z_H=torch.zeros(batch_size, self.num_patches, self.cfg.hidden_size, device=device),
            z_L=torch.zeros(batch_size, self.num_patches, self.cfg.hidden_size, device=device),
        )

    def forward(self, raw: torch.Tensor, state: Optional[WorkingHRMState] = None) -> Dict[str, torch.Tensor]:
        B, L = raw.shape
        assert L == self.cfg.raw_seq_len
        device = raw.device

        patches = raw.view(B, self.num_patches, self.cfg.patch_size)
        pos = self.position_embedding(torch.arange(self.num_patches, device=device)).unsqueeze(0)
        x_input = self.patch_embed(patches) + pos

        # initialize hidden then iterate K times, re-injecting the input each cycle
        # (TRM-style: every cycle sees the original input plus the running state).
        z = torch.zeros_like(x_input)
        for _ in range(self.cfg.num_cycles):
            z = z + x_input
            for layer in self.block:
                z = layer(z, causal_mask=False)

        logits = self.lm_head(self.output_norm(z))
        new_state = self.initial_state(B, device)
        return {"logits": logits, "state": new_state}
