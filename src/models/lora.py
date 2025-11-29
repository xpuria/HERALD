"""
lora (low-rank adaptation) implementation

lora adds trainable low-rank matrices to existing linear layers, allowing
efficient fine-tuning without modifying the original weights.

reference: "lora: low-rank adaptation of large language models" (hu et al., 2021)
https://arxiv.org/abs/2106.09685

this is particularly useful for hrm where most forward passes don't receive gradients
due to the 1-step gradient approximation. lora adapters in the attention layers
that do receive gradients can learn efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LoRALayer(nn.Module):
    """
    lora adaptation layer that adds low-rank updates to a linear layer.
    
    original: w @ x
    with lora: (w + b @ a) @ x = w @ x + (b @ a) @ x
    
    where:
    - w: original frozen weight (d_out x d_in)
    - a: low-rank down projection (r x d_in), initialized with random gaussian
    - b: low-rank up projection (d_out x r), initialized with zeros
    - r: rank (typically 4-16)
    - alpha: scaling factor
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # lora low-rank matrices
        # a is initialized with random gaussian (like kaiming init)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # b is initialized with zeros (so initially lora does nothing)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # initialize lora matrices
        self.reset_parameters()
    
    def reset_parameters(self):
        """initialize lora matrices."""
        # initialize a with kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # b stays at zero (no initial effect)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply lora adaptation.
        
        args:
            x: input tensor of shape (..., in_features)
            
        returns:
            output tensor of shape (..., out_features)
        """
        # compute low-rank update: (x @ a.t) @ b.t
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        # apply dropout and scaling
        lora_output = self.dropout(lora_output) * self.scaling
        
        return lora_output


class LoRALinear(nn.Module):
    """
    linear layer with lora adaptation.
    
    this wraps a standard linear layer and adds lora low-rank adaptation.
    the original weights can be frozen while only training the lora parameters.
    works with both nn.linear and castedlinear.
    """
    
    def __init__(
        self,
        linear: nn.Module,  # can be nn.linear or castedlinear
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        
        # freeze base weights if requested
        if freeze_base:
            for param in self.linear.parameters():
                param.requires_grad = False
        
        # get input/output features from weight shape
        out_features, in_features = linear.weight.shape
        
        # add lora layer
        self.lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass: base linear + lora adaptation.
        
        args:
            x: input tensor of shape (..., in_features)
            
        returns:
            output tensor of shape (..., out_features)
        """
        # base linear transformation
        base_output = self.linear(x)
        
        # add lora adaptation
        lora_output = self.lora(x)
        
        return base_output + lora_output
    
    def merge_weights(self):
        """
        merge lora weights into base weights for inference.
        this eliminates the need for separate lora computation.
        """
        if self.rank > 0:
            # compute low-rank update: b @ a
            lora_weight = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            
            # add to base weight
            self.linear.weight.data += lora_weight
            
            # zero out lora matrices to avoid double-counting
            self.lora.lora_A.data.zero_()
            self.lora.lora_B.data.zero_()


def add_lora_to_linear(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    freeze_base: bool = True,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    recursively add lora to all linear layers in a module.
    
    args:
        module: the module to add lora to
        rank: lora rank
        alpha: lora alpha (scaling factor)
        dropout: lora dropout
        freeze_base: whether to freeze the base weights
        target_modules: list of module name patterns to target (e.g., ['q_proj', 'v_proj'])
                       if none, applies to all linear layers
    
    returns:
        modified module with lora layers
    """
    # import here to avoid circular dependency
    from .layers import CastedLinear
    
    for name, child in module.named_children():
        # check if this layer should be modified (nn.linear or castedlinear)
        should_modify = (
            (isinstance(child, nn.Linear) or isinstance(child, CastedLinear)) and 
            (target_modules is None or any(pattern in name for pattern in target_modules))
        )
        
        if should_modify:
            # replace linear with loralinear
            lora_linear = LoRALinear(
                linear=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                freeze_base=freeze_base
            )
            setattr(module, name, lora_linear)
        else:
            # recursively apply to child modules
            add_lora_to_linear(
                child, 
                rank=rank, 
                alpha=alpha, 
                dropout=dropout, 
                freeze_base=freeze_base,
                target_modules=target_modules
            )
    
    return module


def count_lora_parameters(model: nn.Module) -> dict:
    """
    count lora parameters vs total parameters.
    
    args:
        model: model with lora layers
        
    returns:
        dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
        if 'lora' in name.lower():
            lora_params += num_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lora_params': lora_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'lora_ratio': lora_params / total_params if total_params > 0 else 0
    }


def merge_all_lora_weights(model: nn.Module):
    """
    merge all lora weights into base weights for inference.
    
    args:
        model: model with lora layers
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
