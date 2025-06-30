import numpy as np

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def precision_at_k(y_true, y_pred, k=10):
    """Precision@k: fraction of recommended items in top-k that are relevant."""
    y_true = set(y_true)
    y_pred = y_pred[:k]
    return len(set(y_pred) & y_true) / float(k)

def recall_at_k(y_true, y_pred, k=10):
    """Recall@k: fraction of relevant items that are recommended in top-k."""
    y_true = set(y_true)
    y_pred = y_pred[:k]
    return len(set(y_pred) & y_true) / float(len(y_true) or 1)

def map_at_k(y_true, y_pred, k=10):
    """Mean Average Precision at k."""
    y_true = set(y_true)
    y_pred = y_pred[:k]
    score = 0.0
    hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            hits += 1.0
            score += hits / (i + 1.0)
    return score / min(len(y_true), k) if y_true else 0.0

def ndcg_at_k(y_true, y_pred, k=10):
    """Normalized Discounted Cumulative Gain at k."""
    y_true = set(y_true)
    y_pred = y_pred[:k]
    dcg = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0.0 