import torch
from torch import nn
from .common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Init
        trunc_normal_init_(self.embedding.weight, std=init_std)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs).to(self.cast_to)
