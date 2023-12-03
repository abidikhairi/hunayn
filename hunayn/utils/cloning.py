"""Cloning utilities"""

import copy
from torch import nn


def clones(module: nn.Module, n: int):
    """
    Create a list of identical modules.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): The number of clones to create.

    Returns:
        nn.ModuleList: A list containing `n` identical clones of the input module.

    Example:
        ```python
        encoder_layer = EncoderLayer(d_model=512, d_ff=2048, nhead=8, dropout=0.1)
        cloned_layers = clones(encoder_layer, 6)
        ```

    Note:
        Cloning is performed using `copy.deepcopy()` to ensure that the clones are independent instances.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
