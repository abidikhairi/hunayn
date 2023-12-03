"""Scheduling fucntion"""

def scheduler_fn(step: int, d_model: int, warmup_steps: int):
    """
    Learning Rate Scheduler Function for Transformer Models.

    Args:
        step (int): The current training step.
        d_model (int): The dimension of the model's hidden states.
        warmup_steps (int): The number of warmup steps.

    Returns:
        float: The learning rate for the given step.

    This function calculates the learning rate based on the current training step, model dimension, and the number
    of warmup steps, following the Transformer learning rate schedule.
    """
    step = 1 if step == 0 else step
    return (d_model ** -0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
