import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if not y:
        return 0.0
    proba = {}
    for c in y:
        proba[c] = proba.get(c, 0) + 1.0 / len(y)
    s = -1 * sum( p * np.log2(p) for _, p in proba.items())
    return s
    
    