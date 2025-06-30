import pandas as pd
import os
from .paths import MOVIES_PATH, RATINGS_PATH, USERS_PATH

def load_movies():
    """Load movies.dat as DataFrame with MovieID, Title, Genres."""
    return pd.read_csv(MOVIES_PATH, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

def load_ratings():
    """Load ratings.dat as DataFrame with UserID, MovieID, Rating, Timestamp."""
    return pd.read_csv(RATINGS_PATH, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

def load_users():
    """Load users.dat as DataFrame with UserID, Gender, Age, Occupation, Zip-code."""
    return pd.read_csv(USERS_PATH, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')

def get_user_ratings(ratings_df, user_id):
    """Return DataFrame of (MovieID, Rating) for a given user."""
    return ratings_df[ratings_df['UserID'] == user_id][['MovieID', 'Rating']]

def get_item_features(movies_df):
    """Return list of (MovieID, feature) tuples for LightFM item features (genres)."""
    features = []
    for _, row in movies_df.iterrows():
        genres = [f'genre:{g.strip().lower()}' for g in row['Genres'].split('|')]
        for genre in genres:
            features.append((row['MovieID'], genre))
    return features

def get_user_features(users_df):
    """Return list of (UserID, feature) tuples for LightFM user features (gender, age, occupation)."""
    features = []
    for _, row in users_df.iterrows():
        features.append((row['UserID'], f'gender:{row["Gender"].strip().lower()}'))
        features.append((row['UserID'], f'age:{row["Age"]}'))
        features.append((row['UserID'], f'occupation:{row["Occupation"]}'))
    return features 