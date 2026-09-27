import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A_t = []
    nrow = len(A)
    ncol = len(A[0])
    for j in range(ncol):
        row_t = []
        for i in range(nrow):
            row_t.append(A[i][j])
        A_t.append(row_t)

    return np.array(A_t)