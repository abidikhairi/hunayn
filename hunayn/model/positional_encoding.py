"""Positional encoding module"""
from typing import Tuple
import torch as th
from torch import nn
from xformers.components.positional_embedding import RotaryEmbedding


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer Models.

    Args:
        d_model (int): The dimension of the model's hidden states.

    This class implements positional encoding for Transformer models. It applies positional information
    to the input tensors `q` and `k`.

    Attributes:
        positional_encoding (RotaryEmbedding): The positional encoding module used for adding positional information.
    """

    def __init__(self, d_model: int) -> None:
        """
        Initializes a new PositionalEncoding instance.

        Args:
            d_model (int): The dimension of the model's hidden states.
        """
        super().__init__()
        self.positional_encoding = RotaryEmbedding(dim_model=d_model)

    def forward(self, q: th.Tensor, k: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Applies positional encoding to the input tensors.

        Args:
            q (th.Tensor): The query tensor.
            k (th.Tensor): The key tensor.

        Returns:
            Tuple[th.Tensor, th.Tensor]: The query and key tensors with positional encoding applied.
        """
        q, k = self.positional_encoding(q, k)

        q = q.squeeze(0)
        k = k.squeeze(0)

        return q, k
