import streamlit as st
import pandas as pd
import numpy as np
from movie_recommender.data.loader import load_movies, load_ratings, load_users, get_user_ratings
from movie_recommender.models.content import ContentRecommender
from movie_recommender.models.collaborative import SVDRecommender, SVDppRecommender, NMFRecommender
from movie_recommender.models.hybrid import HybridRecommender
from movie_recommender.utils.metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k, rmse

st.set_page_config(page_title="MovieLens Hybrid Recommender", page_icon="🎬", layout="wide")

# --- Toggle these flags to control retraining and grid search ---
RETRAIN_SVD = False
RETRAIN_SVDPP = False
RETRAIN_NMF = False
TFIDF_PARAMS = {'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0}
ALPHAS = np.linspace(0, 1, 11)

def main():
    """
    Streamlit UI for MovieLens Hybrid Recommender (content-based, SVD, hybrid only).
    """
    st.markdown("""
        <style>
        .main-title {font-size: 2.5rem; font-weight: bold; color: #FF4B4B; margin-bottom: 0.5em;}
        .section-title {font-size: 1.3rem; font-weight: 600; color: #4B8BFF; margin-top: 1.5em;}
        .stButton>button {background-color: #4B8BFF; color: white; font-weight: bold; border-radius: 8px;}
        .stTable {background-color: #f8f9fa;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🎬 MovieLens Hybrid Movie Recommender</div>', unsafe_allow_html=True)
    st.markdown('''<span style="font-size:1.1rem;">Get personalized movie recommendations using content-based, SVD, and hybrid AI models. Select a user or a movie to get started!</span>''', unsafe_allow_html=True)

    @st.cache_data
    def load_all():
        movies = load_movies()
        ratings = load_ratings()
        users = load_users()
        return movies, ratings, users

    movies, ratings, users = load_all()

    with st.sidebar:
        st.markdown('## 👤 User & Movie Selection')
        user_ids = users['UserID'].unique()
        user_id = st.selectbox('Select your User ID', user_ids)
        movie_titles = pd.Series(movies['Title']).sort_values().unique()
        selected_title = st.selectbox('Or select a movie you like:', movie_titles)
        selected_movie_id = movies[movies['Title'] == selected_title].iloc[0]['MovieID']
        st.markdown('---')
        st.markdown('**Tip:** Use the search box to quickly find your favorite movie!')
        st.markdown('---')
        st.markdown('## ⚙️ Model Settings')
        collab_model_type = st.selectbox('Collaborative Model', ['SVD', 'SVD++', 'NMF'])
        retrain_svd = st.checkbox('Retrain SVD', value=RETRAIN_SVD)
        retrain_svdpp = st.checkbox('Retrain SVD++', value=RETRAIN_SVDPP)
        retrain_nmf = st.checkbox('Retrain NMF', value=RETRAIN_NMF)
        st.markdown('---')
        st.markdown('## 🧬 Hybrid Weighting')
        use_grid_alpha = st.checkbox('Tune hybrid alpha (grid search)', value=True)
        use_learned_weights = st.checkbox('Use learned hybrid weights (regression)', value=False)

    @st.cache_resource
    def build_models():
        content_model = ContentRecommender(movies, tfidf_params=TFIDF_PARAMS)
        svd_model = SVDRecommender(ratings, retrain=retrain_svd)
        svdpp_model = SVDppRecommender(ratings, retrain=retrain_svdpp)
        nmf_model = NMFRecommender(ratings, retrain=retrain_nmf)
        return content_model, svd_model, svdpp_model, nmf_model

    content_model, svd_model, svdpp_model, nmf_model = build_models()

    collab_model = {'SVD': svd_model, 'SVD++': svdpp_model, 'NMF': nmf_model}[collab_model_type]

    # Hybrid alpha tuning
    def get_best_alpha():
        best_alpha = 0.6
        best_score = 0.0
        try:
            grid_result = HybridRecommender.grid_search_alpha(content_model, collab_model, ratings, ratings, ALPHAS, k=10)
            if grid_result[0] is not None:
                best_alpha = float(grid_result[0])
                best_score = grid_result[1]
        except Exception:
            pass
        return best_alpha, best_score

    best_alpha, best_score = get_best_alpha() if use_grid_alpha else (0.6, 0.0)
    hybrid_model = HybridRecommender(content_model, collab_model, alpha=best_alpha)
    if use_learned_weights:
        hybrid_model.fit_learned_weights(ratings, ratings, k=10)

    seen_movies = set(ratings[ratings['UserID'] == user_id]['MovieID'])

    st.markdown('<div class="section-title">🔮 Get Recommendations</div>', unsafe_allow_html=True)
    if st.button('Recommend for User'):
        user_ratings = get_user_ratings(ratings, user_id)
        content_recs = content_model.recommend_for_user(user_ratings, top_n=10, exclude_ids=seen_movies)
        collab_movie_ids = collab_model.recommend(user_id, n=10, exclude_ids=seen_movies)
        collab_recs = movies[movies['MovieID'].isin(collab_movie_ids)]
        hybrid_movie_ids = hybrid_model.recommend(user_id, top_n=10, exclude_ids=seen_movies)
        hybrid_recs = movies[movies['MovieID'].isin(hybrid_movie_ids)]

        st.markdown('<div class="section-title">✨ Your Personalized Recommendations</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('#### 🧠 Content-Based (User Profile)')
            st.dataframe(pd.DataFrame(content_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True)
        with col2:
            st.markdown(f'#### 🤖 Collaborative ({collab_model_type})')
            st.dataframe(pd.DataFrame(collab_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True)
        with col3:
            st.markdown('#### 🧬 Hybrid')
            st.dataframe(pd.DataFrame(hybrid_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True)

        # Show metrics for this user
        relevant_movies = set(ratings[(ratings['UserID'] == user_id)]['MovieID'])
        def get_metrics(recs):
            recs = list(recs['MovieID']) if isinstance(recs, pd.DataFrame) else recs
            return {
                'Precision@10': precision_at_k(relevant_movies, recs, k=10),
                'Recall@10': recall_at_k(relevant_movies, recs, k=10),
                'MAP@10': map_at_k(relevant_movies, recs, k=10),
                'NDCG@10': ndcg_at_k(relevant_movies, recs, k=10),
            }
        st.markdown('---')
        st.markdown('### 📊 Metrics for this user')
        st.write('Content-based:', get_metrics(content_recs))
        st.write(f'Collaborative ({collab_model_type}):', get_metrics(collab_recs))
        st.write('Hybrid:', get_metrics(hybrid_recs))

    st.markdown('<div class="section-title">🎥 Recommendations by Movie</div>', unsafe_allow_html=True)
    if selected_title:
        movie_based_recs = content_model.recommend(selected_movie_id, top_n=10)
        st.markdown(f'##### Movies similar to <span style="color:#FF4B4B;">{selected_title}</span>:', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(movie_based_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main() 