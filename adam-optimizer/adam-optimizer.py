import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    m_new = beta1 * np.array(m) + (1-beta1) * np.array(grad)
    m_hat = m_new / (1 - beta1 ** t)
    v_new = beta2 * np.array(v) + (1-beta2) * np.square(np.array(grad))
    v_hat = v_new / (1 - beta2 ** t)
    param_new = np.array(param) - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new