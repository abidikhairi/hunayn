"""Seq2Seq Transformer"""
from typing import Dict, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch as th
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchmetrics.text import Perplexity, BLEUScore
from transformers import PreTrainedTokenizerFast

from hunayn.config import TransformerConfig, OptimizerConfig
from hunayn.model import Embedding, TransformerEncoder, TransformerDecoder
from hunayn.utils.optimizer import scheduler_fn


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

        self.generator = nn.Linear(
            config.d_model, config.tgt_vocab_size, False)

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
            th.Tensor: Output tensor after passing through the Transformer model. 
                Shape: (batch_size, tgt_sequence_length, tgt_vocab_size).
        """
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        memory = self.encoder(src, src_mask)

        z = self.decoder(tgt, memory, tgt_mask, src_mask)

        return self.generator(z)


class HunaynTrainer(pl.LightningModule):
    """
    LightningModule for training the Hunayn model.

    Args:
        model_config (TransformerConfig): Configuration for the Hunayn model.
        optim_config (OptimizerConfig): Configuration for the optimizer.

    Attributes:
        model_config (TransformerConfig): Configuration for the Hunayn model.
        optim_config (OptimizerConfig): Configuration for the optimizer.
        model (Hunayn): Instance of the Hunayn model.
        loss_fn (nn.CrossEntropyLoss): Cross-entropy loss function.
        perplexity (Perplexity): Perplexity measures how well a language model predicts a text sample.
        bleu (BLEUScore): Calculate BLEU score of machine translated text with one or more references.

    Example:
        ```python
        model_config = TransformerConfig(d_model=512, d_ff=2048, nhead=8, dropout=0.1, num_layers=6, 
            src_vocab_size=10000, tgt_vocab_size=10000, src_padding_idx=0, tgt_padding_idx=0)
        optim_config = OptimizerConfig(learning_rate=0.001, beta1=0.9, beta2=0.988, warmup_steps=2000, d_model=512)
        trainer = HunaynTrainer(model_config=model_config, optim_config=optim_config)
        ```

    """
    def __init__(self, model_config: TransformerConfig, optim_config: OptimizerConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.optim_config = optim_config

        self.save_hyperparameters()

        self.model = Hunayn(self.model_config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.perplexity = Perplexity(ignore_index=-100)
        self.blue = BLEUScore(n_gram=4)

        # Need this in test_step
        # self.tgt_tknzr = PreTrainedTokenizerFast.from_pretrained("models/hunayn/english_tokenizer")

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

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Forward pass through the Hunayn model.

        Args:
            src (th.Tensor): Source tensor representing token indices.
            tgt (th.Tensor): Target tensor representing token indices.
            src_mask (th.Tensor): Mask for the source sequence.
            tgt_mask (th.Tensor): Mask for the target sequence.

        Returns:
            th.Tensor: Model output.

        Example:
            ```python
            output = trainer.forward(src_tensor, tgt_tensor, src_mask, tgt_mask)
            ```
        """
        return self.model(src, tgt, src_mask, tgt_mask)

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
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]
        labels = batch["labels"]
        batch_size, seq_len = src.shape

        output = self(src, tgt, src_mask, tgt_mask)

        loss = self.loss_fn(output.view(batch_size * seq_len, -1), labels.view(-1))
        perplexity = self.perplexity(output, labels)

        self.log('train/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('train/perplexity', perplexity, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return {
            "loss": loss
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
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]
        labels = batch["labels"]
        batch_size, seq_len = src.shape

        output = self(src, tgt, src_mask, tgt_mask)

        loss = self.loss_fn(output.view(batch_size * seq_len, -1), labels.view(-1))

        perplexity = self.perplexity(output, labels)

        self.log('valid/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('valid/perplexity', perplexity, prog_bar=True, batch_size=batch_size, sync_dist=True)

        return {
            "loss": loss,
            "perplexity": perplexity
        }

    # @th.no_grad()
    # def test_step(self, batch, batch_idx) -> Dict[str, th.Tensor]:
    #     """
    #     Test step for the Hunayn model.

    #     Args:
    #         batch (Dict): Batch of test data.
    #         batch_idx (int): Batch index.

    #     Returns:
    #         Dict[str, th.Tensor]: Dictionary containing the loss.

    #     Example:
    #         ```python
    #         test_step_output = trainer.test_step(batch, batch_idx)
    #         ```
    #     """
    #     src = batch["src"]
    #     tgt = batch["tgt"]
    #     src_mask = batch["src_mask"]
    #     tgt_mask = batch["tgt_mask"]
    #     labels = batch["labels"]
    #     batch_size, seq_len = src.shape

    #     output = self(src, tgt, src_mask, tgt_mask)

    #     loss = self.loss_fn(output.view(batch_size * seq_len, -1), labels.view(-1))
    #     perplexity = self.perplexity(output, labels)
    #     bleu_score = self._compute_blue_score(output, labels)

    #     self.log('test/loss', loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
    #     self.log('test/perplexity', perplexity, prog_bar=True, batch_size=batch_size, sync_dist=True)
    #     self.log('test/blue_score', bleu_score, prog_bar=True, batch_size=batch_size, sync_dist=True)

    #     return {
    #         "loss": loss
    #     }

    # def _compute_blue_score(self, output: th.Tensor, labels: th.Tensor):
        output = output.argmax(dim=-1)
        preds = []
        tgts = []

        for idx in range(labels.shape[0]):
            row = labels[idx]
            out = output[idx]

            indices = th.argwhere(row != -100).flatten().tolist()
            tgt = row[indices].tolist()
            pred = out[indices].tolist()

            preds.append(self.tgt_tknzr.decode(pred))
            tgts.append(self.tgt_tknzr.decode(tgt))

        return self.blue(preds, tgts)
