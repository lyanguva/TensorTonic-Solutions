import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    new_s = beta * np.array(s) + (1 - beta) * np.square(np.array(g))
    new_w = np.array(w) - lr * np.array(g) / np.sqrt(new_s + eps)

    return new_w, new_s