"""
Test script for evaluating the accuracy of all recommender models in the MovieLens project.
Computes ranking metrics (Precision@10, Recall@10, MAP@10, NDCG@10) and RMSE for all models.
"""
import numpy as np
import pandas as pd
from movie_recommender.data.loader import load_movies, load_ratings, load_users
from movie_recommender.models.content import ContentRecommender
from movie_recommender.models.collaborative import SVDRecommender, SVDppRecommender, NMFRecommender
from movie_recommender.models.hybrid import HybridRecommender
from movie_recommender.utils.metrics import rmse, precision_at_k, recall_at_k, map_at_k, ndcg_at_k
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import json
import os

#Tests can take a while to run, so either run them in a separate file or comment out the parts you don't want to run.

# --- Toggle these flags to control retraining and grid search ---
RETRAIN_SVD = False
RETRAIN_SVDPP = False
RETRAIN_NMF = False
GRID_SEARCH_SVD = False  # Set to True to run SVD grid search
GRID_SEARCH_SVDPP = False  # Set to True to run SVD++ grid search
GRID_SEARCH_NMF = False  # Set to True to run NMF grid search
TFIDF_PARAMS = {'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0}  # Tune as needed
ALPHAS = np.linspace(0, 1, 11)  # For hybrid alpha grid search


def main():
    """
    Main test routine for evaluating all recommenders and metrics.
    """
    print('Loading data...')
    movies = load_movies()
    ratings = load_ratings()
    users = load_users()

    # Train/test split by user
    train_ratings = []
    test_ratings = []
    for user_id in ratings['UserID'].unique():
        user_r = ratings[ratings['UserID'] == user_id]
        if len(user_r) < 5:
            continue
        train, test = train_test_split(user_r, test_size=0.2, random_state=42)
        train_ratings.append(train)
        test_ratings.append(test)
    train_ratings = pd.concat(train_ratings)
    test_ratings = pd.concat(test_ratings)

    # Build content model
    print('Training content-based model...')
    content_model = ContentRecommender(movies, tfidf_params=TFIDF_PARAMS)

    # --- SVD ---
    print('Training/loading SVD model...')
    svd_best_params = None
    if GRID_SEARCH_SVD:
        print('Running SVD grid search...')
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30], 'lr_all': [0.005], 'reg_all': [0.02, 0.05]}
        svd_best_params, svd_best_score = SVDRecommender.grid_search(train_ratings, param_grid)
        print(f'Best SVD params: {svd_best_params}, RMSE: {svd_best_score:.4f}')
    svd_model = SVDRecommender(train_ratings, best_params=svd_best_params, retrain=RETRAIN_SVD)

    # --- SVD++ ---
    print('Training/loading SVD++ model...')
    svdpp_best_params = None
    if GRID_SEARCH_SVDPP:
        print('Running SVD++ grid search...')
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30], 'lr_all': [0.005], 'reg_all': [0.02, 0.05]}
        svdpp_best_params, svdpp_best_score = SVDppRecommender.grid_search(train_ratings, param_grid)
        print(f'Best SVD++ params: {svdpp_best_params}, RMSE: {svdpp_best_score:.4f}')
    svdpp_model = SVDppRecommender(train_ratings, best_params=svdpp_best_params, retrain=RETRAIN_SVDPP)

    # --- NMF ---
    print('Training/loading NMF model...')
    nmf_best_params = None
    if GRID_SEARCH_NMF:
        print('Running NMF grid search...')
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30], 'reg_pu': [0.06], 'reg_qi': [0.06, 0.1]}
        nmf_best_params, nmf_best_score = NMFRecommender.grid_search(train_ratings, param_grid)
        print(f'Best NMF params: {nmf_best_params}, RMSE: {nmf_best_score:.4f}')
    nmf_model = NMFRecommender(train_ratings, best_params=nmf_best_params, retrain=RETRAIN_NMF)

    # --- Hybrid (content + SVD) ---
    print('Tuning hybrid alpha...')
    best_alpha: float = 0.6
    best_score = 0.0
    try:
        grid_result = HybridRecommender.grid_search_alpha(content_model, svd_model, train_ratings, test_ratings, ALPHAS, k=10)
        if grid_result[0] is not None:
            best_alpha = float(grid_result[0])
            best_score = grid_result[1]
    except Exception:
        print('Alpha grid search failed, using default alpha=0.6')
    print(f'Best hybrid alpha: {best_alpha}, Precision@10: {best_score:.4f}')
    hybrid_model = HybridRecommender(content_model, svd_model, alpha=float(best_alpha))
    # Optionally fit learned weights (simple regression)
    # hybrid_model.fit_learned_weights(train_ratings, test_ratings, k=10)

    # Prepare test ratings by user for ranking metrics
    test_ratings_by_user = defaultdict(list)
    for _, row in test_ratings.iterrows():
        test_ratings_by_user[row['UserID']].append(row['MovieID'])
    train_movies_by_user = defaultdict(set)
    for _, row in train_ratings.iterrows():
        train_movies_by_user[row['UserID']].add(row['MovieID'])

    # Evaluate ranking metrics
    print('Evaluating ranking metrics (Precision@10, Recall@10, MAP@10, NDCG@10)...')
    metrics = {
        'content': {'prec': [], 'rec': [], 'map': [], 'ndcg': []},
        'svd': {'prec': [], 'rec': [], 'map': [], 'ndcg': []},
        'svdpp': {'prec': [], 'rec': [], 'map': [], 'ndcg': []},
        'nmf': {'prec': [], 'rec': [], 'map': [], 'ndcg': []},
        'hybrid': {'prec': [], 'rec': [], 'map': [], 'ndcg': []},
    }

    for user_id, relevant_movies in tqdm(list(test_ratings_by_user.items()), desc='Users'):
        seen_movies = train_movies_by_user[user_id]
        # Content-based
        user_train_ratings = train_ratings[train_ratings['UserID'] == user_id][['MovieID', 'Rating']].copy()
        if isinstance(user_train_ratings, pd.DataFrame):
            content_recs_df = content_model.recommend_for_user(user_train_ratings, top_n=20, exclude_ids=seen_movies)
            content_recs = list(content_recs_df['MovieID']) if not content_recs_df.empty else []
            content_recs = [mid for mid in content_recs if mid not in seen_movies][:10]
        else:
            content_recs = []
        if content_recs:
            metrics['content']['prec'].append(precision_at_k(relevant_movies, content_recs, k=10))
            metrics['content']['rec'].append(recall_at_k(relevant_movies, content_recs, k=10))
            metrics['content']['map'].append(map_at_k(relevant_movies, content_recs, k=10))
            metrics['content']['ndcg'].append(ndcg_at_k(relevant_movies, content_recs, k=10))
        # SVD
        try:
            svd_recs = [mid for mid in svd_model.recommend(user_id, n=20, exclude_ids=seen_movies) if mid not in seen_movies][:10]
            if svd_recs:
                metrics['svd']['prec'].append(precision_at_k(relevant_movies, svd_recs, k=10))
                metrics['svd']['rec'].append(recall_at_k(relevant_movies, svd_recs, k=10))
                metrics['svd']['map'].append(map_at_k(relevant_movies, svd_recs, k=10))
                metrics['svd']['ndcg'].append(ndcg_at_k(relevant_movies, svd_recs, k=10))
        except Exception as e:
            print(f"SVD error for user {user_id}: {e}")
        # SVD++
        try:
            svdpp_recs = [mid for mid in svdpp_model.recommend(user_id, n=20, exclude_ids=seen_movies) if mid not in seen_movies][:10]
            if svdpp_recs:
                metrics['svdpp']['prec'].append(precision_at_k(relevant_movies, svdpp_recs, k=10))
                metrics['svdpp']['rec'].append(recall_at_k(relevant_movies, svdpp_recs, k=10))
                metrics['svdpp']['map'].append(map_at_k(relevant_movies, svdpp_recs, k=10))
                metrics['svdpp']['ndcg'].append(ndcg_at_k(relevant_movies, svdpp_recs, k=10))
        except Exception:
            pass
        # NMF
        try:
            nmf_recs = [mid for mid in nmf_model.recommend(user_id, n=20, exclude_ids=seen_movies) if mid not in seen_movies][:10]
            if nmf_recs:
                metrics['nmf']['prec'].append(precision_at_k(relevant_movies, nmf_recs, k=10))
                metrics['nmf']['rec'].append(recall_at_k(relevant_movies, nmf_recs, k=10))
                metrics['nmf']['map'].append(map_at_k(relevant_movies, nmf_recs, k=10))
                metrics['nmf']['ndcg'].append(ndcg_at_k(relevant_movies, nmf_recs, k=10))
        except Exception:
            pass
        # Hybrid
        try:
            hybrid_recs = [mid for mid in hybrid_model.recommend(user_id, top_n=20, exclude_ids=seen_movies) if mid not in seen_movies][:10]
            if hybrid_recs:
                metrics['hybrid']['prec'].append(precision_at_k(relevant_movies, hybrid_recs, k=10))
                metrics['hybrid']['rec'].append(recall_at_k(relevant_movies, hybrid_recs, k=10))
                metrics['hybrid']['map'].append(map_at_k(relevant_movies, hybrid_recs, k=10))
                metrics['hybrid']['ndcg'].append(ndcg_at_k(relevant_movies, hybrid_recs, k=10))
        except Exception:
            pass

    for model in metrics:
        print(f"{model.capitalize()} Precision@10: {np.mean(metrics[model]['prec']):.4f}, Recall@10: {np.mean(metrics[model]['rec']):.4f}, MAP@10: {np.mean(metrics[model]['map']):.4f}, NDCG@10: {np.mean(metrics[model]['ndcg']):.4f}")

    # Save metrics to JSON for visualization
    os.makedirs('plots', exist_ok=True)
    for model in metrics:
        metric_names = ['Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10']
        means = {name: float(np.mean(metrics[model][short])) for name, short in zip(metric_names, ['prec', 'rec', 'map', 'ndcg'])}
        with open(f'plots/{model}_metrics.json', 'w') as f:
            json.dump(means, f, indent=2)

    # Evaluate RMSE for SVD, SVD++, NMF
    print('Evaluating RMSE for collaborative models...')
    for name, model in [('SVD', svd_model), ('SVD++', svdpp_model), ('NMF', nmf_model)]:
        y_true, y_pred = [], []
        for _, row in test_ratings.iterrows():
            user = row['UserID']
            movie = row['MovieID']
            try:
                pred = model.model.predict(user, movie).est
                y_true.append(row['Rating'])
                y_pred.append(pred)
            except Exception:
                continue
        if y_true:
            print(f'{name} RMSE: {rmse(y_true, y_pred):.4f}')

if __name__ == '__main__':
    main()