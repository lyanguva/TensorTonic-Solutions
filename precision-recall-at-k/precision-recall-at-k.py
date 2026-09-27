def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    overlap = len(set(recommended[:k]) & set(relevant))
    return [overlap/k, overlap/len(relevant)]
    