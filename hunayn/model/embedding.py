"""Embedding module"""

import torch as th
from torch import nn

from hunayn.model.positional_encoding import PositionalEncoding


class Embedding(nn.Module):
    """
    Token Embedding Layer for Transformer Models.

    Args:
        vocab_size (int): The size of the vocabulary.
        d_model (int): The dimension of the model's hidden states.
        padding_idx (int): Index for padding tokens.

    This class represents the word embedding layer used in Transformer models, which maps input tokens to
    continuous embeddings and scales them by a factor.

    Attributes:
        embeddings (nn.Embedding): The embedding layer mapping tokens to continuous vectors.
        scale_factor (th.Tensor): The scaling factor for the embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int) -> None:
        """
        Initializes a new EmbeddingLayer instance.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model's hidden states.
        """
        super().__init__()
        self.embeddings = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)
        self.scale_factor = th.tensor(1 / (d_model ** 0.5))

        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=0.6)

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        """
        Applies word embedding and scaling to the input tensor.

        Args:
            inputs (th.Tensor): The input tensor containing token indices.

        Returns:
            th.Tensor: The embedded and scaled tensor.
        """
        x = self.embeddings(inputs) * self.scale_factor
        x = self.positional_encoding(x)
        return x
