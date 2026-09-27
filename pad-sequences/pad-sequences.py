import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.zeros((0, 0), dtype = np.int64)
    if not max_len:
        max_len = max(len(seq) for seq in seqs)
    padded = [
        seq[:max_len] if len(seq) >= max_len else seq + [pad_value] * (max_len - len(seq))
        for seq in seqs 
    ]
    return np.array(padded, dtype = np.int64)