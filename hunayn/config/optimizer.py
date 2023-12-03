"""Optimizer configuration"""

from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    """
    Configuration class for the optimizer.

    Args:
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-3.
        beta1 (float, optional): Exponential decay rate for the first moment estimates (Adam). Default is 0.9.
        beta2 (float, optional): Exponential decay rate for the second moment estimates (Adam). Default is 0.988.
        warmup_steps (int, optional): Number of warm-up steps for learning rate scheduling. Default is 2000.
        d_model (int, optional): Dimensionality of the model's hidden states. Default is 1024.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates (Adam).
        beta2 (float): Exponential decay rate for the second moment estimates (Adam).
        warmup_steps (int): Number of warm-up steps for learning rate scheduling.
        d_model (int): Dimensionality of the model's hidden states.

    Example:
        ```python
        optimizer_config = OptimizerConfig(learning_rate=0.001, beta1=0.9, beta2=0.988, warmup_steps=2000, d_model=1024)
        ```

    """
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.988
    warmup_steps: int = 2000
    d_model: int = 1024
