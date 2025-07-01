"""
Data loading utilities for the MovieLens recommender system.
Handles loading of movies, ratings, and users from the dataset.
"""
import pandas as pd
import os
from .paths import MOVIES_PATH, RATINGS_PATH, USERS_PATH

def load_movies():
    """
    Load movies.dat as a DataFrame with MovieID, Title, Genres.

    Returns:
        pd.DataFrame: Movies data.
    """
    return pd.read_csv(MOVIES_PATH, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')

def load_ratings():
    """
    Load ratings.dat as a DataFrame with UserID, MovieID, Rating, Timestamp.

    Returns:
        pd.DataFrame: Ratings data.
    """
    return pd.read_csv(RATINGS_PATH, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

def load_users():
    """
    Load users.dat as a DataFrame with UserID, Gender, Age, Occupation, Zip-code.

    Returns:
        pd.DataFrame: Users data.
    """
    return pd.read_csv(USERS_PATH, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='ISO-8859-1')

def get_user_ratings(ratings_df, user_id):
    """
    Return DataFrame of (MovieID, Rating) for a given user.

    Args:
        ratings_df (pd.DataFrame): Ratings data.
        user_id (int): User ID.

    Returns:
        pd.DataFrame: User's ratings.
    """
    return ratings_df[ratings_df['UserID'] == user_id][['MovieID', 'Rating']] 