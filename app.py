import os
import sys

# Add the project root to sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from movie_recommender.app import webapp  # noqa: F401
# This file exists so Hugging Face Spaces can use app.py as the entry point.
# The actual Streamlit app is in movie_recommender/app/webapp.py 