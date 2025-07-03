---
title: MovieLens AI Movie Recommender
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🎬 MovieLens AI Movie Recommender

A state-of-the-art hybrid movie recommendation system powered by AI, deployed on Hugging Face Spaces! This application combines collaborative filtering and content-based approaches to deliver personalized movie recommendations using the MovieLens 1M dataset.

## ✨ Features

🤖 **Advanced AI Models**: Hybrid recommendation system combining SVD matrix factorization with TF-IDF content analysis  
🎯 **Personalized Recommendations**: Get tailored movie suggestions based on user preferences  
🔍 **Similar Movie Discovery**: Find movies similar to your favorites  
📊 **High Performance**: Optimized algorithms with proven accuracy metrics  
🚀 **Real-time Processing**: Instant recommendations with cached models  

## 🎮 How to Use

1. **Select a User ID** from the sidebar to get personalized recommendations
2. **Choose a Movie** you like to find similar films
3. **Click the buttons** to generate AI-powered recommendations
4. **Explore** the results and discover your next favorite movie!

## 🧠 AI Technology

- **Hybrid Model**: Combines collaborative and content-based filtering (α=0.6)
- **SVD Matrix Factorization**: Learns user and item patterns from 1M ratings
- **TF-IDF Content Analysis**: Analyzes movie genres, titles, and metadata
- **Performance**: 9.12% Precision@10, 9.81% NDCG@10

## 📊 Dataset

- **MovieLens 1M**: 1 million ratings from 6,000 users on 4,000 movies
- **Rich Metadata**: Genres, titles, release years, user demographics
- **Quality Assured**: Curated dataset widely used in recommender system research

## Features
- **Content-based filtering:** TF-IDF and cosine similarity on genres, titles, and year
- **Collaborative filtering:** SVD, SVD++, NMF (Surprise library)
- **Hybrid model:** Weighted combination of content and collaborative recommenders
- **Streamlit web app:** Clean, client-facing UI for recommendations
- **Advanced metrics:** Precision@10, Recall@10, MAP@10, NDCG@10, RMSE
- **Model persistence:** Save/load models for reproducibility
- **Visualization:** Bar charts, latent factor histograms, and more
- **Testing:** Unified test script for all models and metrics
- **Docker-ready:** Easy containerization and deployment

## Demo

<div align="center">
  <img src="docs/assets/Movies.png" alt="Home Screen" width="45%" />
  <img src="docs/assets/UserID.png" alt="Recommendation Results" width="45%" />
  <br>
  <em>Left: Home screen with movie selection | Right: Personalized recommendations for User ID 1</em>
</div>

## Model Choices & Rationale

This project implements and combines several recommendation models, each chosen for their strengths and complementary properties. **Based on evaluation (MAP@10 and other metrics), SVD was found to be the best collaborative filtering model. The hybrid model used in the app combines content-based filtering with SVD.**

### Content-Based Filtering (TF-IDF + Cosine Similarity)
- **What it is:** Recommends movies based on their metadata (genres, title, year) using TF-IDF vectorization and cosine similarity.
- **Why chosen:** Interpretable, works for new/unrated movies (cold-start), leverages rich metadata, and complements collaborative filtering when user-item data is sparse.
- **Key hyperparameters:** `ngram_range`, `min_df`, `max_df`, `token_pattern` (TF-IDF vectorizer).
- **Evaluation:** Precision@k, Recall@k, MAP@k, NDCG@k (ranking metrics).
- **Strengths:** Cold-start for items, explainable recommendations.
- **Limitations:** Limited by metadata quality, cannot capture collaborative patterns.

### Collaborative Filtering (Matrix Factorization: SVD, SVD++, NMF)
- **What it is:** Learns latent user and item factors from the rating matrix. SVD and NMF decompose the matrix; SVD++ also uses implicit feedback.
- **Why chosen:** State-of-the-art for collaborative filtering, strong performance, interpretable latent factors, robust Surprise library implementations.
- **Key hyperparameters:** `n_factors`, `n_epochs`, `lr_all`, `reg_all`, `reg_pu`, `reg_qi`.
- **Evaluation:** RMSE (rating prediction), Precision@k, Recall@k, MAP@k, NDCG@k (ranking).
- **Strengths:** Captures user-item patterns, good for large/sparse data, factors can be visualized.
- **Limitations:** Cold-start for new users/items, SVD++ is slower, needs sufficient data.
- **Best model:** **SVD** was found to be the best collaborative filtering model in this project, based on MAP@10 and other ranking metrics.

### Hybrid Model (Weighted Content + Collaborative)
- **What it is:** Combines content-based and collaborative scores using a weighted sum (alpha) or learned regression.
- **Why chosen:** Leverages strengths of both approaches, robust to cold-start and sparse data, often outperforms pure models.
- **Key hyperparameters:** `alpha` (content/collaborative weighting), regression weights (optional).
- **Evaluation:** Precision@k, Recall@k, MAP@k, NDCG@k (ranking).
- **Strengths:** Robust, flexible, mitigates weaknesses of individual models.
- **Limitations:** More complex, requires both models, tuning needed.
- **In this project:** The hybrid model used in the app is a combination of the content-based model and the **SVD** collaborative model, as SVD was found to be the best performer.

## Evaluation Metrics

This project uses several standard metrics to evaluate recommendation quality:

- **RMSE (Root Mean Squared Error):**
  - Measures the average difference between predicted and actual ratings.
  - Lower RMSE means better rating prediction accuracy.

- **Precision@10:**
  - Fraction of the top 10 recommended movies that are actually relevant (i.e., the user liked them).
  - High precision means most recommendations are hits.

- **Recall@10:**
  - Fraction of all relevant movies for a user that appear in the top 10 recommendations.
  - High recall means the system finds most of the user's favorites.

- **MAP@10 (Mean Average Precision at 10):**
  - Averages the precision at each position in the top 10 list, considering the order of recommendations.
  - High MAP@10 means relevant movies are ranked higher in the list.

- **NDCG@10 (Normalized Discounted Cumulative Gain at 10):**
  - Measures ranking quality, giving higher scores when relevant movies appear near the top of the list.
  - NDCG is normalized so that 1.0 is the best possible ranking.

## Project Visualizations & Metrics

This project now provides clear, up-to-date visualizations for all recommendation models:

- **Content-Based Model:**
  - Metrics: Precision@10, Recall@10, MAP@10, NDCG@10
  - Visualization: `plots/content_model_metrics.png` (generated by `tools/visualization/visualize_content_model_metrics.py`)

- **Hybrid Model:**
  - Metrics: Precision@10, Recall@10, MAP@10, NDCG@10
  - Visualization: `plots/hybrid_model_metrics.png` (generated by `tools/visualization/visualize_hybrid_model_metrics.py`)

- **Collaborative Filtering Models (SVD, SVD++, NMF):**
  - Metrics: Precision@10, Recall@10, MAP@10, NDCG@10, RMSE
  - Visualization: `plots/all_model_metrics_bar.png` (generated by `tools/visualization/visualize_all_model_metrics.py`)

### How to Generate Metrics and Plots

1. **Run the evaluation script to generate metrics:**
   ```bash
   PYTHONPATH=. python3 tests/test_recommender_accuracy.py
   ```
   This will create JSON files in the `plots/` directory for each model's metrics.

2. **Visualize metrics for each model:**
   - Content-Based:
     ```bash
     python tools/visualization/visualize_content_model_metrics.py
     ```
   - Hybrid:
     ```bash
     python tools/visualization/visualize_hybrid_model_metrics.py
     ```
   - All models (grouped comparison):
     ```bash
     python tools/visualization/visualize_all_model_metrics.py
     ```

3. **Use the generated PNGs in your presentations or reports.**

### Cleaned Plots Directory
- Only the following files are current and relevant:
  - `all_model_metrics_bar.png`
  - `content_metrics.json`
  - `hybrid_metrics.json`
  - `nmf_metrics.json`
  - `svd_metrics.json`
  - `svdpp_metrics.json`

(You may safely delete any other files in `plots/` that are not listed above.)

## Project Structure
```
movie_recommender/
  data/           # Data loading and preprocessing
  models/         # Recommender models (content, collaborative, hybrid)
  app/            # Streamlit web app
  utils/          # Metrics and utilities
plots/            # Output plots
saved_models/     # Serialized model files
ml-1m/            # MovieLens 1M dataset (place here)
tools/
  visualization/  # Visualization scripts for model evaluation
    visualize_all_models.py
    visualize_best_model.py
tests/            # Unified test script
main.py           # Entrypoint (CLI)
requirements.txt  # Python dependencies
```

## Docker Usage

### 1. Build the Docker Image
From the project root (where the Dockerfile is located), run:
```bash
docker build -t movie-recommender .
```

### 2. Run the Docker Container
To start the Streamlit app and map port 8501:
```bash
docker run -p 8501:8501 movie-recommender
```
The app will be available at [http://localhost:8501](http://localhost:8501).

- To run a shell inside the container (for debugging or manual commands):
  ```bash
  docker run -it --rm movie-recommender bash
  ```

- To run a specific script (e.g., main.py):
  ```bash
  docker run -it --rm movie-recommender python main.py
  ```

### 3. Notes
- Make sure the `ml-1m` dataset is present in the project root before building the image.
- You can customize the default command in the Dockerfile if you want the container to always start the app.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- Place the `ml-1m` folder in the project root (next to `main.py`).
- Download from [GroupLens](https://grouplens.org/datasets/movielens/1m/).

## Usage

### Run the Streamlit Web App
```bash
PYTHONPATH=. streamlit run movie_recommender/app/webapp.py
```
- Or use the CLI:
```bash
python main.py --app
```

### Train or Evaluate Models (CLI)
```bash
python main.py --train      # Train all models
python main.py --evaluate   # Evaluate all models
```

### Run All Tests
```bash
python tests/test_recommender_accuracy.py
```

### Visualize Model Performance
```bash
python tools/visualization/visualize_all_models.py
python tools/visualization/visualize_best_model.py
```
- Plots are saved in the `plots/` folder with descriptive filenames.

## Key Modules
- **data/loader.py:** Data loading utilities
- **data/paths.py:** Robust, portable path management
- **models/content.py:** Content-based recommender (TF-IDF/cosine)
- **models/collaborative.py:** SVD, SVD++, NMF recommenders (Surprise)
- **models/hybrid.py:** Hybrid recommender (weighted content + collaborative)
- **utils/metrics.py:** Advanced evaluation metrics
- **app/webapp.py:** Streamlit UI (client-facing, hybrid-only)
- **tests/test_recommender_accuracy.py:** Unified test script for all models/metrics
- **tools/visualization/:** Visualization scripts for model evaluation

## Deploying on Hugging Face Spaces

You can deploy this project as an interactive demo on [Hugging Face Spaces](https://huggingface.co/spaces) using Streamlit.

### Steps:
1. **Prepare your repository:**
   - Ensure all code and `requirements.txt` are in the root or appropriate subfolders.
   - The main app entry point is: `movie_recommender/app/webapp.py` (Streamlit app).
   - If needed, you can symlink or copy this file to `app.py` in the root for Spaces compatibility.

2. **Data considerations:**
   - Spaces has a 15GB limit. If your dataset is large, consider using a smaller sample or downloading data at runtime.
   - Place any required data in the repo or add code to download it in the app.

3. **Deploy:**
   - Create a new Space on Hugging Face (choose Streamlit as the SDK).
   - Upload your code or link your GitHub repo.
   - Set the entry point to `movie_recommender/app/webapp.py` (or `app.py` if you moved it).
   - Wait for the build and your app will be live!

### Local Testing
To test locally before deploying:
```bash
streamlit run movie_recommender/app/webapp.py
```

---
