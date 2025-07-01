"""
Main entry point for the MovieLens Hybrid Recommender System.
Supports training, evaluation, and launching the Streamlit web app.
"""
import argparse
from movie_recommender.data.loader import load_movies, load_ratings, load_users
from movie_recommender.models.content import ContentRecommender
from movie_recommender.models.collaborative import CollaborativeRecommender
from movie_recommender.models.hybrid import HybridRecommender
from movie_recommender.utils.metrics import rmse
import numpy as np
import streamlit.web.bootstrap
import os

def train_models():
    """
    Train all models (content-based, collaborative, hybrid).

    Returns:
        tuple: (content_model, collab_model, hybrid_model)
    """
    print('Loading data...')
    movies = load_movies()
    ratings = load_ratings()
    print('Training content-based model...')
    content_model = ContentRecommender(movies)
    print('Training collaborative model...')
    collab_model = CollaborativeRecommender(ratings)
    interactions = collab_model.prepare()
    collab_model.train(interactions)
    print('Training hybrid model...')
    hybrid_model = HybridRecommender(content_model, collab_model)
    print('Training complete.')
    return content_model, collab_model, hybrid_model

def evaluate_models():
    """
    Evaluate all models using RMSE and print results.
    """
    print('Evaluating models...')
    movies = load_movies()
    ratings = load_ratings()
    content_model = ContentRecommender(movies)
    collab_model = CollaborativeRecommender(ratings)
    interactions = collab_model.prepare()
    collab_model.train(interactions)
    hybrid_model = HybridRecommender(content_model, collab_model)
    # Simple RMSE evaluation for collaborative filtering
    y_true = []
    y_pred = []
    for _, row in ratings.sample(1000, random_state=42).iterrows():
        user = row['UserID']
        movie = row['MovieID']
        if user in collab_model.user_id_map and movie in collab_model.item_id_map:
            pred = collab_model.model.predict(collab_model.user_id_map[user], np.array([collab_model.item_id_map[movie]]))[0]
            y_true.append(row['Rating'])
            y_pred.append(pred)
    print(f'Collaborative Filtering RMSE (LightFM raw score): {rmse(y_true, y_pred):.4f}')
    print('Evaluation complete.')

def run_app():
    """
    Launch the Streamlit web application.
    """
    print('Launching Streamlit app...')
    app_path = os.path.join(os.path.dirname(__file__), 'movie_recommender/app/webapp.py')
    streamlit.web.bootstrap.run(app_path, '', [], {})

def main():
    """
    Parse command-line arguments and run the appropriate mode.
    """
    parser = argparse.ArgumentParser(description='MovieLens Hybrid Recommender System')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--app', action='store_true', help='Run web app')
    args = parser.parse_args()

    if args.train:
        train_models()
    if args.evaluate:
        evaluate_models()
    if args.app:
        run_app()

if __name__ == '__main__':
    main() 