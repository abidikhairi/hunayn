"""Encoder module"""

from typing import  Union, Dict
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch as th
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from hunayn.model.positional_encoding import PositionalEncoding
from hunayn.model.embedding import Embedding
from hunayn.config import EncoderConfig, OptimizerConfig
from hunayn.utils.cloning import clones
from hunayn.utils.optimizer import scheduler_fn


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
            nn.GELU(),
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
        q = x
        k = x
        v= x

        # pre-normalization
        v = self.attn_norm(v)
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

        self.layers = clones(EncoderLayer(d_model, d_ff, nhead, dropout), num_layers)

    def forward(self, x: th.Tensor, mask: Union[th.BoolTensor, th.Tensor] = None) -> th.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (th.Tensor): Input tensor. Shape: (batch_size, sequence_length, d_model).
            mask (Union[th.BoolTensor, th.Tensor], optional): Attention mask tensor to mask padded tokens.
                Shape: (batch_size, sequence_length, sequence_length).

        Returns:
            th.Tensor: Output tensor after passing through the Transformer Encoder. 
                Shape: (batch_size, sequence_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x

class HunaynEncoder(nn.Module):
    """
    HunaynEncoder class represents the encoder module of a Transformer-based model.

    Attributes:
        embedding (nn.Embedding): Embedding layer mapping token IDs to continuous vectors.
        encoder (TransformerEncoder): Transformer encoder layer for processing sequences.

    Methods:
        forward(x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
            Defines the forward pass of the encoder.

    """
    def __init__(self, model_config: EncoderConfig) -> None:
        """
        Initialize the HunaynEncoder module.

        Args:
            model_config (EncoderConfig): Configuration object for the encoder.
        """
        super().__init__()

        self.embedding = Embedding(
            model_config.src_vocab_size, model_config.d_model, model_config.src_padding_idx)

        self.encoder = TransformerEncoder(
            num_layers=model_config.num_encoder_layers,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            dropout=model_config.dropout,
            nhead=model_config.nhead
        )

    def forward(self, x: th.Tensor, mask: th.Tensor = None) -> th.Tensor:
        """
        Forward pass of the HunaynEncoder.

        Args:
            x (th.Tensor): Input tensor containing token IDs.
            mask (th.Tensor, optional): Optional tensor for masking.

        Returns:
            th.Tensor: Encoded output tensor.
        """
        x = self.embedding(x)
        return self.encoder(x, mask)

class HunaynEncoderTrainer(pl.LightningModule):
    """
    HunaynEncoderTrainer is a PyTorch Lightning Module representing the encoder component of the Hunayn model.

    Attributes:
        model_config (EncoderConfig): Configuration for the transformer model.
        optim_config (OptimizerConfig): Configuration for the optimizer.
        encoder (TransformerEncoder): Transformer encoder module.
        loss_fn (nn.CrossEntropyLoss): Binary Cross-entropy loss function.
    """
    def __init__(self, model_config: EncoderConfig, optim_config: OptimizerConfig):
        """
        Initializes the HunaynEncoder.

        Args:
            model_config (TransformerConfig): Configuration for the transformer model.
            optim_config (OptimizerConfig): Configuration for the optimizer.
        """
        super().__init__()

        self.model_config = model_config
        self.optim_config = optim_config

        self.save_hyperparameters()

        self.encoder = HunaynEncoder(model_config)

        self.predictor = nn.Sequential(
            nn.Linear(model_config.d_model, 1, False),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x: th.Tensor, mask: th.Tensor = None):
        """
        Forward pass of the HunaynEncoder.

        Args:
            x (th.Tensor): Input tensor representing token indices.
            mask (th.Tensor, optional): Mask tensor for masking padded tokens during processing.

        Returns:
            th.Tensor: Output tensor from the predictor.
        """
        return self.encoder(x, mask)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[Optimizer, LRScheduler]: Optimizer and learning rate scheduler.

        Example:
            ```python
            optimizer, scheduler = trainer.configure_optimizers()
            ```
        """
        optimizer = optim.AdamW(
            self.parameters(), lr=self.optim_config.learning_rate, betas=(self.optim_config.beta1, self.optim_config.beta2))
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda step: scheduler_fn(step, self.optim_config.d_model, self.optim_config.warmup_steps))
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx) -> Dict[str, th.Tensor]:
        """
        Training step for the Hunayn model.

        Args:
            batch (Dict): Batch of training data.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, th.Tensor]: Dictionary containing the loss.

        Example:
            ```python
            training_step_output = trainer.training_step(batch, batch_idx)
            ```
        """
        x = batch['x']
        mask = batch['mask']
        y = batch['y'].float().view(-1)
        y_mask = batch['y_mask'].view(-1)
        batch_size = x.size(0)

        h = self.predictor(self(x, mask))
        y_pred = h.view(-1)

        loss = self.loss_fn(y_pred[y_mask], y[y_mask])

        self.log('train/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

        return {
            'loss': loss
        }

    @th.no_grad()
    def validation_step(self, batch, batch_idx) -> Dict[str, th.Tensor]:
        """
        Validation step for the Hunayn model.

        Args:
            batch (Dict): Batch of validation data.
            batch_idx (int): Batch index.

        Returns:
            Dict[str, th.Tensor]: Dictionary containing the loss.

        Example:
            ```python
            validation_step_output = trainer.validation_step(batch, batch_idx)
            ```
        """
        x = batch['x']
        mask = batch['mask']
        y = batch['y'].float().view(-1)
        y_mask = batch['y_mask'].view(-1)
        batch_size = x.size(0)

        h = self.predictor(self(x, mask))
        y_pred = h.view(-1)

        loss = self.loss_fn(y_pred[y_mask], y[y_mask])

        self.log('valid/loss', loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

        return {
            'loss': loss,
            "valid/loss": loss
        }
