"""
Content-based recommender model for the MovieLens system.

Model: Content-Based Filtering (TF-IDF + Cosine Similarity)
----------------------------------------------------------
This module implements a content-based movie recommender using TF-IDF vectorization and cosine similarity on movie genres, titles, and release year.

- **How it works:**
  Each movie is represented as a text feature vector (combining genres, title, and year). TF-IDF transforms these into numerical vectors, and cosine similarity is used to find movies most similar to a given movie or a user's top-rated movie.
- **Why this approach:**
  Content-based filtering is interpretable, does not require user history for new items (solves cold-start for items), and leverages rich metadata. It complements collaborative filtering, especially when user-item interactions are sparse.
- **Key hyperparameters:**
  - `ngram_range`, `min_df`, `max_df`, and `token_pattern` for TF-IDF vectorization (see constructor).
- **Evaluation:**
  Evaluated using ranking metrics: Precision@k, Recall@k, MAP@k, NDCG@k, which measure the relevance and ranking quality of recommendations.
- **Strengths:**
  - Works for new/unrated movies (cold-start)
  - Recommendations are explainable (based on content)
- **Limitations:**
  - Limited by the quality and granularity of metadata
  - Cannot capture collaborative/user preference patterns
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Optional, Set

class ContentRecommender:
    """
    Content-based recommender using TF-IDF on genres, titles, and year, with cosine similarity.

    This model recommends movies based on their content features (genres, title, year) by computing TF-IDF vectors and using cosine similarity. It is especially useful for cold-start scenarios and provides interpretable recommendations.

    Args:
        movies_df (pd.DataFrame): DataFrame of movies.
        tfidf_params (dict, optional): Parameters for TfidfVectorizer.
    """
    def __init__(self, movies_df: pd.DataFrame, tfidf_params: Optional[dict] = None):
        self.movies_df = movies_df.copy()
        # Extract year from title using regex
        self.movies_df['Year'] = self.movies_df['Title'].apply(lambda x: re.findall(r'\((\d{4})\)', x))
        self.movies_df['Year'] = self.movies_df['Year'].apply(lambda x: x[0] if x else '')
        # Combine genres, title, and year for richer content features
        self.movies_df['content'] = self.movies_df['Genres'] + ' ' + self.movies_df['Title'] + ' ' + self.movies_df['Year']
        default_params = {'token_pattern': r'[^|\s]+', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0}
        if tfidf_params:
            default_params.update(tfidf_params)
        self.tfidf = TfidfVectorizer(**default_params)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['content'])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_id: int, top_n: int = 10, exclude_ids: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Recommend top-N similar movies to a given movie.

        Args:
            movie_id (int): Movie ID to find similarities for.
            top_n (int): Number of recommendations.
            exclude_ids (set, optional): Movie IDs to exclude.

        Returns:
            pd.DataFrame: Top-N recommended movies. Empty DataFrame if movie_id not found.
        """
        idx_list = self.movies_df.index[self.movies_df['MovieID'] == movie_id].tolist()
        if not idx_list:
            # Edge case: movie_id not found
            return pd.DataFrame()
        idx = idx_list[0]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in sim_scores[1:] if (exclude_ids is None or self.movies_df.iloc[i]['MovieID'] not in exclude_ids)]
        return self.movies_df.iloc[top_indices[:top_n]]

    def recommend_for_user(self, user_ratings: pd.DataFrame, top_n: int = 10, exclude_ids: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Recommend movies for a user based on their highest-rated movie.

        Args:
            user_ratings (pd.DataFrame): User's ratings.
            top_n (int): Number of recommendations.
            exclude_ids (set, optional): Movie IDs to exclude.

        Returns:
            pd.DataFrame: Top-N recommended movies.
        """
        if user_ratings.empty:
            return pd.DataFrame()
        top_movie_id = user_ratings.sort_values('Rating', ascending=False).iloc[0]['MovieID']
        seen = set(user_ratings['MovieID']) if exclude_ids is None else set(user_ratings['MovieID']).union(set(exclude_ids))
        return self.recommend(top_movie_id, top_n=top_n, exclude_ids=seen)

    def recommend_by_title(self, title: str, top_n: int = 10, exclude_ids: Optional[Set[int]] = None) -> pd.DataFrame:
        """
        Recommend movies similar to a given title (fuzzy match).

        Args:
            title (str): Movie title to search for.
            top_n (int): Number of recommendations.
            exclude_ids (set, optional): Movie IDs to exclude.

        Returns:
            pd.DataFrame: Top-N recommended movies. Empty DataFrame if no match found.
        """
        matches = self.movies_df[self.movies_df['Title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        movie_id = matches.iloc[0]['MovieID']
        return self.recommend(movie_id, top_n=top_n, exclude_ids=exclude_ids) 