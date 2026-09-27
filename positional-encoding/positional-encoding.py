import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    i_max = np.ceil(d_model / 2)
    # create result with d+1 as buffer
    pe = np.zeros((seq_len, d_model + 1))
    i = np.arange(0, i_max, 1, dtype=np.int_)
    pos = np.arange(0, seq_len, 1, dtype=np.int_)
    pe[:, 2*i] = np.sin( pos[:, np.newaxis] / base**(2*i/d_model) )
    pe[:, 2*i + 1] = np.cos( pos[:, np.newaxis] / base**(2*i/d_model) )

    return pe[:, :d_model]
    