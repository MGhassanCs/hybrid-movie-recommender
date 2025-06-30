# MovieLens 1M Hybrid Movie Recommendation System

A modular Python project for building a hybrid movie recommender system using the MovieLens 1M dataset. Combines content-based (genres) and collaborative filtering (user ratings) with a web interface (Streamlit/Gradio).

## Features
- Content-based filtering using genres (TF-IDF, cosine similarity)
- Collaborative filtering using user ratings (LightFM/Surprise)
- Hybrid recommendations
- Web app for interactive recommendations

## Project Structure
```
movie_recommender/
  data/           # Data loading and preprocessing
  models/         # Recommender models
  app/            # Web app
  utils/          # Utilities (metrics, etc.)
  main.py         # Entrypoint
```

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `python main.py` or `streamlit run app/webapp.py`

## Dataset
Place the `ml-1m` folder in the project root. No need to download again if already present.

## Authors
- Your Name 