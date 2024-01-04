"""Positional encoding module"""
import math
import torch as th
from torch import nn


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module adds positional encodings to the input embeddings.
    These encodings are added to give the model information about the position of tokens within a sequence.

    Attributes:
        dropout (nn.Dropout): Dropout layer for regularization.
        pe (th.Tensor): Positional encodings.

    Methods:
        forward(x: th.Tensor) -> th.Tensor:
            Adds positional encodings to the input tensor and returns the result.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of the model.
            dropout (float, optional): Dropout rate. Default is 0.1.
            max_len (int, optional): Maximum length of the input sequence. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(35000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Output tensor with positional encodings added.
        """
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)
