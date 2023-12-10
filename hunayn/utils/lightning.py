"""Lightning utils"""
import os
from pytorch_lightning import loggers as pl_loggers, callbacks as pl_callbacks


def lightning_loggers():
    """
    Initialize and return a list of Lightning loggers.

    Returns:
        List[pl_loggers.BaseLogger]: A list of Lightning loggers, including CSVLogger and WandbLogger if enabled.

    Notes:
        - CSVLogger logs training and validation metrics to a CSV file.
        - WandbLogger logs metrics to the Weights & Biases service if the 'USE_WANDB' environment variable is set.

    Example:
        ```python
        # Example usage:
        # loggers = lightning_loggers()
        # trainer = pl.Trainer(logger=loggers)
        # ```
    """
    loggers = []
    loggers.append(pl_loggers.CSVLogger('logs', name="hunayn", version=1))

    if "USE_WANDB" in os.environ:
        loggers.append(pl_loggers.WandbLogger(project="hunayn", experiment="toasty-dragon-11"))

    return loggers

def lightning_callbacks():
    """
    Initialize and return a list of Lightning callbacks.

    Returns:
        List[pl_callbacks.Callback]: A list of Lightning callbacks, including ModelCheckpoint.

    Notes:
        - ModelCheckpoint saves the model's checkpoints during training based on specified conditions.

    Example:
        ```python
        # Example usage:
        # callbacks = lightning_callbacks()
        # trainer = pl.Trainer(callbacks=callbacks)
        # ```
    """
    callbacks = []
    callbacks.append(pl_callbacks.ModelCheckpoint(
        dirpath="models/checkpoints", filename="hunayn_v1", monitor="valid/loss"))

    return callbacks
