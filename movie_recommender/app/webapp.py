"""
Streamlit web application for the MovieLens Hybrid Movie Recommender.
Client-facing UI for personalized movie recommendations using a hybrid AI model.
"""
import streamlit as st
import pandas as pd
import numpy as np
from movie_recommender.data.loader import load_movies, load_ratings, load_users, get_user_ratings
from movie_recommender.models.content import ContentRecommender
from movie_recommender.models.collaborative import SVDRecommender
from movie_recommender.models.hybrid import HybridRecommender
from movie_recommender.utils.metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k

st.set_page_config(page_title="MovieLens AI Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_all():
    """
    Load all required data (movies, ratings, users) for the app.

    Returns:
        tuple: (movies, ratings, users) DataFrames.
    """
    movies = load_movies()
    ratings = load_ratings()
    users = load_users()
    return movies, ratings, users

movies, ratings, users = load_all()

@st.cache_resource
def build_models():
    """
    Build and cache the content and hybrid recommender models.

    Returns:
        tuple: (content_model, hybrid_model)
    """
    content_model = ContentRecommender(movies)
    svd_model = SVDRecommender(ratings, retrain=False)
    # Use a fixed alpha (e.g., 0.6) as best found
    hybrid_model = HybridRecommender(content_model, svd_model, alpha=0.6)
    return content_model, hybrid_model

content_model, hybrid_model = build_models()

# --- UI Styling and Layout ---
st.markdown("""
    <style>
    .main-title {font-size: 2.5rem; font-weight: bold; color: #FF4B4B; margin-bottom: 0.5em;}
    .section-title {font-size: 1.3rem; font-weight: 600; color: #4B8BFF; margin-top: 1.5em;}
    .stButton>button {background-color: #4B8BFF; color: white; font-weight: bold; border-radius: 8px;}
    .stTable {background-color: #f8f9fa;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 MovieLens AI Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('''<span style="font-size:1.1rem;">Get personalized movie recommendations powered by a state-of-the-art AI hybrid model, combining collaborative and content-based intelligence for the best results. Select your user or a movie to get started!</span>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 👤 User & Movie Selection')
    user_ids = users['UserID'].unique()
    user_id = st.selectbox('Select your User ID', user_ids)
    movie_titles = pd.Series(movies['Title']).sort_values().unique()
    selected_title = st.selectbox('Or select a movie you like:', movie_titles)
    selected_movie_id = movies[movies['Title'] == selected_title].iloc[0]['MovieID']
    st.markdown('---')
    st.markdown('**Tip:** Use the search box to quickly find your favorite movie!')

# Track movies already seen by the user
seen_movies = set(ratings[ratings['UserID'] == user_id]['MovieID'])

st.markdown('<div class="section-title">🔮 Your Personalized Recommendations</div>', unsafe_allow_html=True)
if st.button('Recommend for Me!'):
    user_ratings = get_user_ratings(ratings, user_id)
    hybrid_movie_ids = hybrid_model.recommend(user_id, top_n=10, exclude_ids=seen_movies)
    hybrid_recs = movies[movies['MovieID'].isin(hybrid_movie_ids)]

    st.markdown('<div class="section-title">✨ Your Top Movie Picks</div>', unsafe_allow_html=True)
    st.write("Here are some movies we think you'll love! Sit back, relax, and enjoy your personalized recommendations.")
    st.dataframe(pd.DataFrame(hybrid_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">🎥 Find Similar Movies</div>', unsafe_allow_html=True)
if selected_title:
    movie_based_recs = content_model.recommend(selected_movie_id, top_n=10)
    st.markdown(f'##### Movies similar to <span style="color:#FF4B4B;">{selected_title}</span>:', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(movie_based_recs)[['Title', 'Genres']].reset_index(drop=True), use_container_width=True, hide_index=True) 