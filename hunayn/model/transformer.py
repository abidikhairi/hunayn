"""Seq2Seq Transformer"""
from typing import Union
import torch as th
from torch import nn

from hunayn.config import TransformerConfig
from hunayn.model import Embedding, TransformerEncoder, TransformerDecoder


class Hunayn(nn.Module):
    """
    Transformer model for protein translation.

    Args:
        config (TransformerConfig): Configuration for the Transformer model.

    Attributes:
        src_embedding (Embedding): Embedding layer for the source sequence.
        tgt_embedding (Embedding): Embedding layer for the target sequence.
        encoder (TransformerEncoder): Transformer Encoder module.
        decoder (TransformerDecoder): Transformer Decoder module.
        generator (nn.Linear): Transformer Generator (LM Head).

    Example:
        ```python
        transformer = Transformer(config=my_transformer_config)
        output = transformer(source_tensor, target_tensor, src_mask, tgt_mask)
        ```

    """
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.src_embedding = Embedding(
            config.src_vocab_size, config.d_model, config.src_padding_idx)
        self.tgt_embedding = Embedding(
            config.tgt_vocab_size, config.d_model, config.tgt_padding_idx)

        self.encoder = TransformerEncoder(
            config.num_encoder_layers, config.d_model, config.d_ff, config.nhead, config.dropout)
        self.decoder = TransformerDecoder(
            config.num_decoder_layers, config.d_model, config.d_ff, config.nhead, config.dropout)

        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size, False)

    def forward(self, src: th.Tensor, tgt: th.Tensor,
                src_mask: Union[th.Tensor, th.BoolTensor] = None, tgt_mask: Union[th.Tensor, th.BoolTensor] = None) -> th.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            src (th.Tensor): Source tensor representing token indices. Shape: (batch_size, src_sequence_length).
            tgt (th.Tensor): Target tensor representing token indices. Shape: (batch_size, tgt_sequence_length).
            src_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the source sequence.
                Shape: (batch_size, src_sequence_length).
            tgt_mask (Union[th.Tensor, th.BoolTensor], optional): Attention mask for the target sequence.
                Shape: (batch_size, tgt_sequence_length).

        Returns:
            th.Tensor: Output tensor after passing through the Transformer model. Shape: (batch_size, tgt_sequence_length, tgt_vocab_size).
        """
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        memory = self.encoder(src, src_mask)

        z = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.generator(z)
