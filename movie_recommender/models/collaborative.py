"""
Collaborative filtering models for the MovieLens recommender system.

Models: SVD, SVD++, NMF (Matrix Factorization)
---------------------------------------------
This module implements collaborative filtering recommenders using matrix factorization techniques:
- **SVD (Singular Value Decomposition):** Decomposes the user-item rating matrix into user and item latent factors. Predicts ratings as the dot product of these factors.
- **SVD++:** Extends SVD by incorporating implicit feedback (e.g., which items a user has interacted with, not just explicit ratings).
- **NMF (Non-negative Matrix Factorization):** Similar to SVD but constrains factors to be non-negative, which can improve interpretability.

- **Why these models:**
  Matrix factorization models are state-of-the-art for collaborative filtering, offering strong performance and interpretability. The Surprise library provides robust, well-tested implementations.
- **Key hyperparameters:**
  - `n_factors`: Number of latent factors (dimensionality)
  - `n_epochs`: Number of training epochs
  - `lr_all`, `reg_all`, `reg_pu`, `reg_qi`: Learning rate and regularization
- **Evaluation:**
  Evaluated using RMSE (for rating prediction) and ranking metrics: Precision@k, Recall@k, MAP@k, NDCG@k (for top-N recommendation quality).
- **Strengths:**
  - Captures complex user-item interaction patterns
  - Good for large, sparse datasets
  - Latent factors can be visualized/interpreted
- **Limitations:**
  - Cold-start problem for new users/items
  - Requires sufficient data for each user/item
  - SVD++ is slower to train than SVD/NMF
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import SVD, SVDpp, NMF, Dataset as SurpriseDataset, Reader
from sklearn.model_selection import ParameterGrid
import joblib
import os
from typing import Optional

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

class SVDRecommender:
    """
    SVD collaborative recommender using Surprise library (matrix factorization).

    - **How it works:** Learns user and item latent factors from the rating matrix. Predicts ratings as the dot product of these factors.
    - **Why SVD:** Strong baseline for collaborative filtering, efficient, interpretable.
    - **Key hyperparameters:** n_factors, n_epochs, lr_all, reg_all.
    - **Evaluation:** RMSE, Precision@k, Recall@k, MAP@k, NDCG@k.
    - **Strengths:** Good for dense data, interpretable factors.
    - **Limitations:** Cold-start for new users/items, does not use implicit feedback.
    """
    def __init__(self, ratings_df: pd.DataFrame, model_path: Optional[str] = None, best_params: Optional[dict] = None, retrain: bool = False):
        self.ratings_df = ratings_df
        self.model = None
        self.trainset = None
        self.user_items = defaultdict(set)
        self.model_path = model_path or os.path.join(MODEL_DIR, 'svd_model.pkl')
        self.best_params = best_params
        if not retrain and os.path.exists(self.model_path):
            self.load()
        else:
            self.train()

    def train(self) -> None:
        """
        Train the SVD model on the ratings data.
        """
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(self.ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        self.trainset = data.build_full_trainset()
        params = self.best_params or {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.02}
        self.model = SVD(**params)
        self.model.fit(self.trainset)
        # Build user-items mapping for fast exclusion
        for _, row in self.ratings_df.iterrows():
            self.user_items[row['UserID']].add(row['MovieID'])
        self.save()

    def recommend(self, user_id: int, n: int = 10, exclude_ids: Optional[set] = None) -> list:
        """
        Recommend top-N movies for a user, excluding already seen movies.

        Args:
            user_id (int): User ID.
            n (int): Number of recommendations.
            exclude_ids (set, optional): Movie IDs to exclude.

        Returns:
            list: Top-N recommended movie IDs. Empty list if user not found.
        """
        if self.model is None or self.trainset is None or user_id not in self.user_items:
            return []
        all_inner_ids = set(self.trainset.all_items())
        all_movie_ids = set(self.trainset.to_raw_iid(iid) for iid in all_inner_ids)
        seen = self.user_items[user_id] if exclude_ids is None else self.user_items[user_id].union(set(exclude_ids))
        candidates = all_movie_ids - seen
        predictions = [(iid, self.model.predict(user_id, iid).est) for iid in candidates]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in predictions[:n]]

    def save(self) -> None:
        """
        Save the trained model, trainset, and user-items mapping to disk.
        """
        joblib.dump((self.model, self.trainset, self.user_items), self.model_path)

    def load(self) -> None:
        """
        Load the model, trainset, and user-items mapping from disk.
        """
        self.model, self.trainset, self.user_items = joblib.load(self.model_path)

    @staticmethod
    def grid_search(ratings_df: pd.DataFrame, param_grid: dict) -> tuple:
        """
        Perform grid search for SVD hyperparameters.

        Args:
            ratings_df (pd.DataFrame): Ratings data.
            param_grid (dict): Grid of hyperparameters.

        Returns:
            tuple: (best_params, best_score)
        """
        from surprise.model_selection import GridSearchCV
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        return gs.best_params['rmse'], gs.best_score['rmse']

class SVDppRecommender:
    """
    SVD++ collaborative recommender using Surprise library (matrix factorization with implicit feedback).

    - **How it works:** Extends SVD by incorporating implicit feedback (which items a user has interacted with, not just ratings).
    - **Why SVD++:** Improves accuracy by leveraging more user behavior data.
    - **Key hyperparameters:** n_factors, n_epochs, lr_all, reg_all.
    - **Evaluation:** RMSE, Precision@k, Recall@k, MAP@k, NDCG@k.
    - **Strengths:** Handles implicit feedback, better for sparse data.
    - **Limitations:** Slower to train, still cold-start for new users/items.
    """
    def __init__(self, ratings_df, model_path=None, best_params=None, retrain=False):
        self.ratings_df = ratings_df
        self.model = None
        self.trainset = None
        self.user_items = defaultdict(set)
        self.model_path = model_path or os.path.join(MODEL_DIR, 'svdpp_model.pkl')
        self.best_params = best_params
        if not retrain and os.path.exists(self.model_path):
            self.load()
        else:
            self.train()

    def train(self):
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(self.ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        self.trainset = data.build_full_trainset()
        params = self.best_params or {'n_factors': 50, 'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.02}
        self.model = SVDpp(**params)
        self.model.fit(self.trainset)
        for _, row in self.ratings_df.iterrows():
            self.user_items[row['UserID']].add(row['MovieID'])
        self.save()

    def recommend(self, user_id, n=10, exclude_ids=None):
        if self.model is None or self.trainset is None:
            raise ValueError('Model not trained.')
        all_inner_ids = set(self.trainset.all_items())
        all_movie_ids = set(self.trainset.to_raw_iid(iid) for iid in all_inner_ids)
        seen = self.user_items[user_id] if exclude_ids is None else self.user_items[user_id].union(set(exclude_ids))
        candidates = all_movie_ids - seen
        predictions = [(iid, self.model.predict(user_id, iid).est) for iid in candidates]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in predictions[:n]]

    def save(self):
        joblib.dump((self.model, self.trainset, self.user_items), self.model_path)

    def load(self):
        self.model, self.trainset, self.user_items = joblib.load(self.model_path)

    @staticmethod
    def grid_search(ratings_df, param_grid):
        from surprise.model_selection import GridSearchCV
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        return gs.best_params['rmse'], gs.best_score['rmse']

class NMFRecommender:
    """
    NMF collaborative recommender using Surprise library (non-negative matrix factorization).

    - **How it works:** Factorizes the rating matrix into non-negative user/item factors. Can improve interpretability.
    - **Why NMF:** Useful when non-negativity is desired, interpretable factors.
    - **Key hyperparameters:** n_factors, n_epochs, reg_pu, reg_qi.
    - **Evaluation:** RMSE, Precision@k, Recall@k, MAP@k, NDCG@k.
    - **Strengths:** Non-negative factors, interpretable.
    - **Limitations:** Cold-start, may underperform SVD/SVD++ on some data.
    """
    def __init__(self, ratings_df, model_path=None, best_params=None, retrain=False):
        self.ratings_df = ratings_df
        self.model = None
        self.trainset = None
        self.user_items = defaultdict(set)
        self.model_path = model_path or os.path.join(MODEL_DIR, 'nmf_model.pkl')
        self.best_params = best_params
        if not retrain and os.path.exists(self.model_path):
            self.load()
        else:
            self.train()

    def train(self):
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(self.ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        self.trainset = data.build_full_trainset()
        params = self.best_params or {'n_factors': 100, 'n_epochs': 30, 'reg_pu': 0.06, 'reg_qi': 0.06}
        self.model = NMF(**params)
        self.model.fit(self.trainset)
        for _, row in self.ratings_df.iterrows():
            self.user_items[row['UserID']].add(row['MovieID'])
        self.save()

    def recommend(self, user_id, n=10, exclude_ids=None):
        if self.model is None or self.trainset is None:
            raise ValueError('Model not trained.')
        all_inner_ids = set(self.trainset.all_items())
        all_movie_ids = set(self.trainset.to_raw_iid(iid) for iid in all_inner_ids)
        seen = self.user_items[user_id] if exclude_ids is None else self.user_items[user_id].union(set(exclude_ids))
        candidates = all_movie_ids - seen
        predictions = [(iid, self.model.predict(user_id, iid).est) for iid in candidates]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in predictions[:n]]

    def save(self):
        joblib.dump((self.model, self.trainset, self.user_items), self.model_path)

    def load(self):
        self.model, self.trainset, self.user_items = joblib.load(self.model_path)

    @staticmethod
    def grid_search(ratings_df, param_grid):
        from surprise.model_selection import GridSearchCV
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        gs = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        return gs.best_params['rmse'], gs.best_score['rmse'] 