"""
vq-vae: vector quantized variational autoencoder

compresses financial time series into discrete tokens by learning
a codebook of 512 pattern templates. encoder compresses 64 timesteps
to 16 tokens (4x reduction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """maps continuous encoder outputs to nearest codebook entries"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # squared distances to codebook
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # vq losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices.view(input_shape[0], -1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, 
                     kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=hidden_channels, out_channels=in_channels, 
                     kernel_size=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    """compresses 64 timesteps to 16 positions (4x downsampling)"""
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        
        self._conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=num_hiddens // 2, 
                                 kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2, out_channels=num_hiddens, 
                                 kernel_size=4, stride=2, padding=1)
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        
    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x


class Decoder(nn.Module):
    """reconstructs signal from quantized representation (training only)"""
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=num_hiddens, 
                                 kernel_size=3, padding=1)
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        
        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens, out_channels=num_hiddens // 2, 
                                                kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2, out_channels=out_channels, 
                                                kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


class VQVAE(nn.Module):
    """
    complete vq-vae: encode -> quantize -> decode
    after training, only encode + quantize is used for tokenization
    """
    
    def __init__(self, num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32, 
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, 1)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        
        return loss, x_recon, perplexity, encoding_indices
