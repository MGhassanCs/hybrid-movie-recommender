import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import SVD, SVDpp, NMF, Dataset as SurpriseDataset, Reader
from sklearn.model_selection import ParameterGrid
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

class SVDRecommender:
    """
    SVD collaborative recommender using Surprise library, with grid search and model saving/loading.
    """
    def __init__(self, ratings_df, model_path=None, best_params=None, retrain=False):
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

    def train(self):
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(self.ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        self.trainset = data.build_full_trainset()
        params = self.best_params or {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.02}
        self.model = SVD(**params)
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
        """
        Perform grid search for SVD hyperparameters. Returns best params and score.
        """
        from surprise.model_selection import GridSearchCV
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(ratings_df[['UserID', 'MovieID', 'Rating']], reader)
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
        gs.fit(data)
        return gs.best_params['rmse'], gs.best_score['rmse']

class SVDppRecommender:
    """
    SVD++ collaborative recommender using Surprise library, with grid search and model saving/loading.
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
        params = self.best_params or {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.02}
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
    NMF collaborative recommender using Surprise library, with grid search and model saving/loading.
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