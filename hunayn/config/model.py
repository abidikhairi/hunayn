"""Main model configuration"""

from pydantic import BaseModel


class TransformerConfig(BaseModel):
    """
        Configuration class for a Transformer model.

        Args:
            d_model (int, optional): Dimensionality of the model's hidden states (default is 1024).
            d_ff (int, optional): Dimensionality of the feedforward layer in the model (default is 512).
            nhead (int, optional): Number of attention heads in the model (default is 4).
            dropout (float, optional): Dropout probability applied to various parts of the model (default is 0.6).
            num_encoder_layers (int, optional): Number of encoder layers in the model (default is 4).
            num_decoder_layers (int, optional): Number of decoder layers in the model (default is 4).
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            src_padding_idx (int): Padding index for the source sequences.
            tgt_padding_idx (int): Padding index for the target sequences.

        Note:
            Ensure that the vocabulary sizes and padding indices are appropriate for your specific task.

        Attributes:
            d_model (int): Dimensionality of the model's hidden states.
            d_ff (int): Dimensionality of the feedforward layer in the model.
            nhead (int): Number of attention heads in the model.
            dropout (float): Dropout probability applied to various parts of the model.
            num_encoder_layers (int): Number of encoder layers in the model.
            num_decoder_layers (int): Number of decoder layers in the model.
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            src_padding_idx (int): Padding index for the source sequences.
            tgt_padding_idx (int): Padding index for the target sequences.
    """
    d_model: int = 1024
    d_ff: int = 512
    nhead: int = 4
    dropout: float = 0.6
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    src_vocab_size: int
    tgt_vocab_size: int
    src_padding_idx: int
    tgt_padding_idx: int


class TrainerConfig(BaseModel):
    """
    Configuration class for the training process.

    Attributes:
        batch_size (int, optional): Batch size for training. Defaults to 4.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.

    Example:
        ```python
        trainer_config = TrainerConfig(batch_size=8, num_workers=6)
        ```
    """
    batch_size: int = 4
    num_workers: int = 4
