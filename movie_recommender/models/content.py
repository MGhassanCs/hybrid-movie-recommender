import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class ContentRecommender:
    """
    Content-based recommender using TF-IDF on genres, titles, and year, with cosine similarity.
    Allows TF-IDF parameter tuning.
    """
    def __init__(self, movies_df, tfidf_params=None):
        self.movies_df = movies_df.copy()
        # Extract year from title
        self.movies_df['Year'] = self.movies_df['Title'].apply(lambda x: re.findall(r'\\((\\d{4})\\)', x))
        self.movies_df['Year'] = self.movies_df['Year'].apply(lambda x: x[0] if x else '')
        # Combine genres, title, and year for richer content features
        self.movies_df['content'] = self.movies_df['Genres'] + ' ' + self.movies_df['Title'] + ' ' + self.movies_df['Year']
        default_params = {'token_pattern': r'[^|\\s]+', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0}
        if tfidf_params:
            default_params.update(tfidf_params)
        self.tfidf = TfidfVectorizer(**default_params)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['content'])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_id, top_n=10, exclude_ids=None):
        idx = self.movies_df.index[self.movies_df['MovieID'] == movie_id][0]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in sim_scores[1:] if (exclude_ids is None or self.movies_df.iloc[i]['MovieID'] not in exclude_ids)]
        return self.movies_df.iloc[top_indices[:top_n]]

    def recommend_for_user(self, user_ratings, top_n=10, exclude_ids=None):
        if user_ratings.empty:
            return pd.DataFrame()
        top_movie_id = user_ratings.sort_values('Rating', ascending=False).iloc[0]['MovieID']
        seen = set(user_ratings['MovieID']) if exclude_ids is None else set(user_ratings['MovieID']).union(set(exclude_ids))
        return self.recommend(top_movie_id, top_n=top_n, exclude_ids=seen)

    def recommend_by_title(self, title, top_n=10, exclude_ids=None):
        matches = self.movies_df[self.movies_df['Title'].str.contains(title, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        movie_id = matches.iloc[0]['MovieID']
        return self.recommend(movie_id, top_n=top_n, exclude_ids=exclude_ids) 