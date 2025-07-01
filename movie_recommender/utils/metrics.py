"""
Evaluation metrics for recommender systems.
Includes RMSE, Precision@k, Recall@k, MAP@k, and NDCG@k.
"""
import numpy as np

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        y_true (list or np.ndarray): True ratings.
        y_pred (list or np.ndarray): Predicted ratings.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def precision_at_k(y_true, y_pred, k=10):
    """
    Compute Precision@k: fraction of recommended items in top-k that are relevant.

    Args:
        y_true (list): Relevant item IDs.
        y_pred (list): Predicted item IDs (ordered).
        k (int): Top-k cutoff.

    Returns:
        float: Precision@k value.
    """
    y_true = set(y_true)
    y_pred = y_pred[:k]
    return len(set(y_pred) & y_true) / float(k)

def recall_at_k(y_true, y_pred, k=10):
    """
    Compute Recall@k: fraction of relevant items that are recommended in top-k.

    Args:
        y_true (list): Relevant item IDs.
        y_pred (list): Predicted item IDs (ordered).
        k (int): Top-k cutoff.

    Returns:
        float: Recall@k value.
    """
    y_true = set(y_true)
    y_pred = y_pred[:k]
    return len(set(y_pred) & y_true) / float(len(y_true) or 1)

def map_at_k(y_true, y_pred, k=10):
    """
    Compute Mean Average Precision at k (MAP@k).

    Args:
        y_true (list): Relevant item IDs.
        y_pred (list): Predicted item IDs (ordered).
        k (int): Top-k cutoff.

    Returns:
        float: MAP@k value.
    """
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
    """
    Compute Normalized Discounted Cumulative Gain at k (NDCG@k).

    Args:
        y_true (list): Relevant item IDs.
        y_pred (list): Predicted item IDs (ordered).
        k (int): Top-k cutoff.

    Returns:
        float: NDCG@k value.
    """
    y_true = set(y_true)
    y_pred = y_pred[:k]
    dcg = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0.0 