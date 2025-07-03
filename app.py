import os
import sys
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

# Set surprise data folder to a writable location
os.environ['SURPRISE_DATA_FOLDER'] = '/tmp/surprise_data'

# Add the project root to sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our models and data loaders
from movie_recommender.data.loader import load_movies, load_ratings, load_users
from movie_recommender.models.content import ContentRecommender
from movie_recommender.models.collaborative import SVDRecommender
from movie_recommender.models.hybrid import HybridRecommender

# Configure Streamlit page
st.set_page_config(
    page_title="🎬 MovieLens AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_data
def get_user_ratings(ratings: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Get ratings for a specific user.
    
    Args:
        ratings (pd.DataFrame): Ratings DataFrame.
        user_id (int): User ID.
        
    Returns:
        pd.DataFrame: User's ratings.
    """
    return ratings[ratings['UserID'] == user_id][['MovieID', 'Rating']]

# Load data once
try:
    movies, ratings, users = load_all()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

@st.cache_resource
def build_models():
    """
    Build and cache the content and hybrid recommender models.
    
    Returns:
        tuple: (content_model, hybrid_model)
    """
    try:
        content_model = ContentRecommender(movies)
        svd_model = SVDRecommender(ratings, retrain=False)
        # Use a fixed alpha (e.g., 0.6) as best found
        hybrid_model = HybridRecommender(content_model, svd_model, alpha=0.6)
        return content_model, hybrid_model
    except Exception as e:
        st.error(f"Error building models: {e}")
        return None, None

# Build models once
content_model, hybrid_model = build_models()

if content_model is None or hybrid_model is None:
    st.error("Failed to load recommendation models. Please check the data and model files.")
    st.stop()

# --- UI Styling and Layout ---
st.markdown("""
    <style>
    .main-title {font-size: 2.5rem; font-weight: bold; color: #FF4B4B; margin-bottom: 0.5em;}
    .section-title {font-size: 1.3rem; font-weight: 600; color: #4B8BFF; margin-top: 1.5em;}
    .stButton>button {
        background-color: #4B8BFF; 
        color: white; 
        font-weight: bold; 
        border-radius: 8px;
        width: 100%;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3A7BD5;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stTable {background-color: #f8f9fa;}
    .movie-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4B8BFF;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🎬 MovieLens AI Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('''<span style="font-size:1.1rem;">Get personalized movie recommendations powered by a state-of-the-art AI hybrid model, combining collaborative and content-based intelligence for the best results. Select your user or a movie to get started!</span>''', unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown('## 👤 User & Movie Selection')
    
    # User selection
    user_ids = sorted(users['UserID'].unique())
    user_id = st.selectbox('Select your User ID', user_ids, help="Choose a user ID to get personalized recommendations")
    
    # Movie selection
    movie_titles = sorted(movies['Title'].unique())
    selected_title = st.selectbox('Or select a movie you like:', movie_titles, help="Find movies similar to this one")
    selected_movie_id = movies[movies['Title'] == selected_title].iloc[0]['MovieID']
    
    st.markdown('---')
    st.markdown('### 📊 Model Information')
    st.info('''
    **Hybrid AI Model**:
    - Content-based filtering
    - Collaborative filtering (SVD)
    - Optimized combination (α=0.6)
    ''')
    
    st.markdown('**💡 Tip:** Use the search box to quickly find your favorite movie!')

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-title">🔮 Personalized Recommendations</div>', unsafe_allow_html=True)
    
    if st.button('🎯 Get My Recommendations', key="personal_recs"):
        with st.spinner('🤖 AI is analyzing your preferences...'):
            try:
                # Track movies already seen by the user
                seen_movies = set(ratings[ratings['UserID'] == user_id]['MovieID'])
                
                # Get hybrid recommendations
                hybrid_movie_ids = hybrid_model.recommend(user_id, top_n=10, exclude_ids=seen_movies)
                
                if hybrid_movie_ids:
                    hybrid_recs = movies[movies['MovieID'].isin(hybrid_movie_ids)]
                    
                    st.success(f"🎉 Found {len(hybrid_recs)} perfect matches for User {user_id}!")
                    
                    # Display recommendations in a nice format
                    for idx, (_, movie) in enumerate(hybrid_recs.iterrows(), 1):
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{idx}. {movie['Title']}</h4>
                            <p><strong>Genres:</strong> {movie['Genres']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found for this user. Try a different user ID.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

with col2:
    st.markdown('<div class="section-title">🎥 Similar Movies</div>', unsafe_allow_html=True)
    
    if st.button('🔍 Find Similar Movies', key="similar_movies"):
        with st.spinner('🔍 Finding movies like this one...'):
            try:
                movie_based_recs = content_model.recommend(selected_movie_id, top_n=10)
                
                if not movie_based_recs.empty:
                    st.success(f"🎬 Found {len(movie_based_recs)} movies similar to '{selected_title}'!")
                    
                    # Display similar movies
                    for idx, (_, movie) in enumerate(movie_based_recs.iterrows(), 1):
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{idx}. {movie['Title']}</h4>
                            <p><strong>Genres:</strong> {movie['Genres']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No similar movies found. Try a different movie.")
                    
            except Exception as e:
                st.error(f"Error finding similar movies: {e}")

# Footer with additional information
st.markdown('---')
st.markdown('### 📈 About This Recommender')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🤖 AI Technology**
    - Hybrid recommendation system
    - SVD matrix factorization
    - TF-IDF content analysis
    - Optimized performance metrics
    """)

with col2:
    st.markdown("""
    **📊 Dataset**
    - MovieLens 1M dataset
    - 1 million ratings
    - 6,000 users
    - 4,000 movies
    """)

with col3:
    st.markdown("""
    **🎯 Performance**
    - Precision@10: 9.12%
    - Recall@10: 3.42%
    - MAP@10: 4.30%
    - NDCG@10: 9.81%
    """)

st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.9em;">
    💡 Built with Streamlit • Powered by MovieLens Data • AI Hybrid Recommendation System
</div>
""", unsafe_allow_html=True)
