"""Encoder module"""

from typing import Union
import torch as th
from torch import nn

from hunayn.utils.cloning import clones
from hunayn.config.model import DecoderConfig
from hunayn.model.embedding import Embedding

class DecoderLayer(nn.Module):
    """
    Decoder layer module for a Transformer model.

    Args:
        d_model (int): Dimensionality of the model's hidden states.
        d_ff (int): Dimensionality of the feedforward layer in the model.
        nhead (int): Number of attention heads in the model.
        dropout (float): Dropout probability applied to various parts of the model.

    Attributes:
        self_attn (nn.MultiheadAttention): Multi-head self-attention mechanism.
        cross_attn (nn.MultiheadAttention): Multi-head attention mechanism to attend to the encoder's output.
        feedforward (nn.Sequential): Feedforward neural network module.
        attn_norm (nn.LayerNorm): Layer normalization for the self-attention output.
        cross_attn_norm (nn.LayerNorm): Layer normalization for the cross-attention output.
        feedforward_norm (nn.LayerNorm): Layer normalization for the feedforward output.

    Example:
        ```python
        decoder_layer = DecoderLayer(d_model=512, d_ff=2048, nhead=8, dropout=0.1)
        output = decoder_layer(target_tensor, source_tensor, tgt_mask, src_mask)
        ```

    """
    def __init__(self, d_model: int, d_ff: int, nhead: int, dropout: float) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.attn_norm = nn.LayerNorm(d_model)  # change to RMSNorm
        self.cross_attn_norm = nn.LayerNorm(d_model)  # change to RMSNorm
        self.feedforward_norm = nn.LayerNorm(d_model)  # change to RMSNorm

    def forward(self, tgt: th.Tensor, src: th.Tensor,
                tgt_mask: Union[th.Tensor, th.BoolTensor] = None,
                src_mask: Union[th.Tensor, th.BoolTensor] = None) -> th.Tensor:
        """
        Forward pass through the DecoderLayer.

        Args:
            tgt (th.Tensor): Target tensor representing token indices. Shape: (batch_size, tgt_sequence_length, d_model).
            src (th.Tensor): Source tensor representing token indices. Shape: (batch_size, src_sequence_length, d_model).
            tgt_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the target sequence.
                Shape: (batch_size, tgt_sequence_length).
            src_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the source sequence.
                Shape: (batch_size, src_sequence_length).

        Returns:
            th.Tensor: Output tensor after passing through the DecoderLayer.
                Shape: (batch_size, tgt_sequence_length, d_model).
        """
        q = tgt
        k = tgt
        v = tgt

        v = self.attn_norm(tgt)
        q = self.attn_norm(q)
        k = self.attn_norm(k)

        z, _ = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              need_weights=False, is_causal=True)
        tgt = tgt + z  # skip-connection (good when backpropagating gradients)

        tgt = self.cross_attn_norm(tgt)
        z, _ = self.cross_attn(src, src, tgt, attn_mask=src_mask,
                              need_weights=False, is_causal=False)
        tgt = tgt + z

        tgt = self.feedforward_norm(tgt)
        tgt = tgt + self.feedforward(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder module composed of multiple layers.

    Args:
        num_layers (int): Number of decoder layers.
        d_model (int): Dimensionality of the model's hidden states.
        d_ff (int): Dimensionality of the feedforward layer in each decoder layer.
        nhead (int): Number of attention heads in each decoder layer.
        dropout (float): Dropout probability applied to various parts of each decoder layer.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.

    Example:
        ```python
        decoder = TransformerDecoder(num_layers=6, d_model=512, d_ff=2048, nhead=8, dropout=0.1)
        output = decoder(target_tensor, source_tensor, tgt_mask, src_mask)
        ```

    """
    def __init__(self, num_layers: int, d_model: int, d_ff: int, nhead: int, dropout: float) -> None:
        super().__init__()

        self.layers = clones(DecoderLayer(d_model, d_ff, nhead, dropout), num_layers)

    def forward(self, tgt: th.Tensor, src: th.Tensor,
                tgt_mask: Union[th.Tensor, th.BoolTensor] = None,
                src_mask: Union[th.Tensor, th.BoolTensor] = None) -> th.Tensor:
        """
        Forward pass through the Transformer Decoder.

        Args:
            tgt (th.Tensor): Target tensor representing token indices. Shape: (batch_size, tgt_sequence_length, d_model).
            src (th.Tensor): Source tensor representing token indices. Shape: (batch_size, src_sequence_length, d_model).
            tgt_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the target sequence.
                Shape: (batch_size, tgt_sequence_length).
            src_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the source sequence.
                Shape: (batch_size, src_sequence_length).

        Returns:
            th.Tensor: Output tensor after passing through the Transformer Decoder.
                Shape: (batch_size, tgt_sequence_length, d_model).
        """
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return tgt


class HunaynDecoder(nn.Module):
    def __init__(self, decoder_config: DecoderConfig) -> None:
        super().__init__()

        self.embedding = Embedding(vocab_size=decoder_config.tgt_vocab_size, d_model=decoder_config.d_model,
                                   padding_idx=decoder_config.tgt_padding_idx)

        self.decoder = TransformerDecoder(num_layers=decoder_config.num_decoder_layers, d_model=decoder_config.d_model,
                                          d_ff=decoder_config.d_ff, nhead=decoder_config.nhead, dropout=decoder_config.dropout)

    def forward(self, tgt: th.Tensor, z: th.Tensor,
                tgt_mask: Union[th.Tensor, th.BoolTensor] = None,
                src_mask: Union[th.Tensor, th.BoolTensor] = None):

        x = self.embedding(tgt)
        x = self.decoder(x, z, tgt_mask, src_mask)
        
        return x
