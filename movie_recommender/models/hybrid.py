import numpy as np
from sklearn.linear_model import LinearRegression

class HybridRecommender:
    """
    Hybrid recommender: weighted combination of content and collaborative recommenders.
    Supports alpha grid search and optionally a learned combination.
    """
    def __init__(self, content_model, collaborative_model, alpha=0.6):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha  # weight for content-based, (1-alpha) for collaborative
        self.lr_model = None

    def recommend(self, user_id, top_n=10, exclude_ids=None):
        # Get collaborative recommendations (movie IDs and scores)
        collab_movie_ids = self.collaborative_model.recommend(user_id, n=100, exclude_ids=exclude_ids)
        # For each, get content similarity score to user's top-rated movie
        user_ratings = self.collaborative_model.ratings_df
        user_movies = user_ratings[user_ratings['UserID'] == user_id]
        if user_movies.empty:
            return []
        top_movie_id = user_movies.sort_values('Rating', ascending=False).iloc[0]['MovieID']
        content_scores = {}
        for mid in collab_movie_ids:
            try:
                sim = self.content_model.sim_matrix[
                    self.content_model.movies_df.index[self.content_model.movies_df['MovieID'] == top_movie_id][0],
                    self.content_model.movies_df.index[self.content_model.movies_df['MovieID'] == mid][0]
                ]
            except:
                sim = 0
            content_scores[mid] = sim
        # Combine scores (collaborative is just rank, content is similarity)
        hybrid_scores = {}
        for rank, mid in enumerate(collab_movie_ids):
            collab_score = 1 - (rank / len(collab_movie_ids))  # higher rank = higher score
            if self.lr_model is not None:
                # Use learned regression model
                X = np.array([[content_scores[mid], collab_score]])
                hybrid_scores[mid] = self.lr_model.predict(X)[0]
            else:
                hybrid_scores[mid] = self.alpha * content_scores[mid] + (1 - self.alpha) * collab_score
        # Return top_n movie IDs, filter out seen
        top_movies = [mid for mid, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)]
        if exclude_ids is not None:
            top_movies = [mid for mid in top_movies if mid not in exclude_ids]
        return top_movies[:top_n]

    @staticmethod
    def grid_search_alpha(content_model, collaborative_model, ratings_df, test_ratings_df, alphas, k=10):
        """
        Grid search for best alpha on validation set using Precision@k.
        """
        from movie_recommender.utils.metrics import precision_at_k
        best_alpha = None
        best_score = -1
        for alpha in alphas:
            model = HybridRecommender(content_model, collaborative_model, alpha=alpha)
            scores = []
            test_ratings_by_user = test_ratings_df.groupby('UserID')['MovieID'].apply(list)
            train_movies_by_user = ratings_df.groupby('UserID')['MovieID'].apply(set)
            for user_id, relevant_movies in test_ratings_by_user.items():
                seen_movies = train_movies_by_user.get(user_id, set())
                recs = model.recommend(user_id, top_n=k, exclude_ids=seen_movies)
                if recs:
                    scores.append(precision_at_k(recs, relevant_movies, k=k))
            avg_score = np.mean(scores) if scores else 0
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        return best_alpha, best_score

    def fit_learned_weights(self, ratings_df, test_ratings_df, k=10):
        """
        Fit a linear regression to combine content and collaborative scores for top-k recommendations.
        """
        X, y = [], []
        test_ratings_by_user = test_ratings_df.groupby('UserID')['MovieID'].apply(list)
        train_movies_by_user = ratings_df.groupby('UserID')['MovieID'].apply(set)
        for user_id, relevant_movies in test_ratings_by_user.items():
            seen_movies = train_movies_by_user.get(user_id, set())
            collab_movie_ids = self.collaborative_model.recommend(user_id, n=100, exclude_ids=seen_movies)
            user_ratings = self.collaborative_model.ratings_df
            user_movies = user_ratings[user_ratings['UserID'] == user_id]
            if user_movies.empty:
                continue
            top_movie_id = user_movies.sort_values('Rating', ascending=False).iloc[0]['MovieID']
            for rank, mid in enumerate(collab_movie_ids[:k]):
                try:
                    sim = self.content_model.sim_matrix[
                        self.content_model.movies_df.index[self.content_model.movies_df['MovieID'] == top_movie_id][0],
                        self.content_model.movies_df.index[self.content_model.movies_df['MovieID'] == mid][0]
                    ]
                except:
                    sim = 0
                collab_score = 1 - (rank / len(collab_movie_ids))
                X.append([sim, collab_score])
                y.append(1 if mid in relevant_movies else 0)
        if X and y:
            self.lr_model = LinearRegression().fit(X, y) 