"""Encoder module"""

from typing import Union
import torch as th
from torch import nn

from hunayn.model.positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Encoder layer module for a Transformer model.

    Args:
        d_model (int): Dimensionality of the model's hidden states.
        d_ff (int): Dimensionality of the feedforward layer in the model.
        nhead (int): Number of attention heads in the model.
        dropout (int): Dropout probability applied to various parts of the model.

    Attributes:
        pe (PositionalEncoding): Positional encoding module for introducing positional information.
        self_attn (nn.MultiheadAttention): Multi-head self-attention mechanism.
        feedforward (nn.Sequential): Feedforward neural network module.
        attn_norm (nn.LayerNorm): Layer normalization for the attention output.
        feedforward_norm (nn.LayerNorm): Layer normalization for the feedforward output.

    Example:
        ```python
        encoder_layer = EncoderLayer(d_model=512, d_ff=2048, nhead=8, dropout=0.1)
        output = encoder_layer(input_tensor, attn_mask)
        ```

    """

    def __init__(self, d_model: int, d_ff: int, nhead: int, dropout: int) -> None:
        super().__init__()

        self.pe = PositionalEncoding(d_model=d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.attn_norm = nn.LayerNorm(d_model)  # change to RMSNorm
        self.feedforward_norm = nn.LayerNorm(d_model)  # change to RMSNorm

    def forward(self, x: th.Tensor, mask: Union[th.BoolTensor, th.Tensor] = None) -> th.Tensor:
        """
        Applies the forward pass of the encoder layer.

        Args:
            x (th.Tensor): The input tensor.
            mask (th.Tensor): The input mask.
        """
        query_pe, key_pe = self.pe(x, x)
        q = x + query_pe
        k = x + key_pe

        # pre-normalization
        v = self.attn_norm(x)
        q = self.attn_norm(q)
        k = self.attn_norm(k)

        z, _ = self.self_attn(q, k, v, attn_mask=mask,
                              need_weights=False, is_causal=False)
        x = x + z  # skip-connection (good when backpropagating gradients)

        # pre-normalization
        x = self.feedforward_norm(x)
        x = x + self.feedforward(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module composed of multiple layers.

    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): Dimensionality of the model's hidden states.
        d_ff (int): Dimensionality of the feedforward layer in each encoder layer.
        nhead (int): Number of attention heads in each encoder layer.
        dropout (float): Dropout probability applied to various parts of each encoder layer.

    Attributes:
        layers (nn.ModuleList): List of encoder layers.

    Example:
        ```python
        encoder = TransformerEncoder(num_layers=6, d_model=512, d_ff=2048, nhead=8, dropout=0.1)
        output = encoder(input_tensor, attention_mask)
        ```

    """
    def __init__(self, num_layers: int, d_model: int, d_ff: int, nhead: int, dropout: float) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: th.Tensor, mask: Union[th.BoolTensor, th.Tensor] = None) -> th.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (th.Tensor): Input tensor. Shape: (batch_size, sequence_length, d_model).
            mask (Union[th.BoolTensor, th.Tensor], optional): Attention mask tensor to mask padded tokens.
                Shape: (batch_size, sequence_length, sequence_length).

        Returns:
            th.Tensor: Output tensor after passing through the Transformer Encoder. Shape: (batch_size, sequence_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)

        return x
