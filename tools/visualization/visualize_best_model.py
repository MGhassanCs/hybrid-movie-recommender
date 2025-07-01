"""
Visualization script for the best collaborative model in the MovieLens recommender system.
Generates and saves bar charts and latent factor histograms for the best model by MAP@10.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from movie_recommender.data.loader import load_movies, load_ratings
from movie_recommender.utils.metrics import precision_at_k, recall_at_k, map_at_k, ndcg_at_k
from sklearn.model_selection import train_test_split

MODEL_DIR = 'saved_models'
MODELS = {
    'SVD': os.path.join(MODEL_DIR, 'svd_model.pkl'),
    'SVD++': os.path.join(MODEL_DIR, 'svdpp_model.pkl'),
    'NMF': os.path.join(MODEL_DIR, 'nmf_model.pkl'),
}

movies = load_movies()
ratings = load_ratings()

# Split ratings into train/test by user (80/20)
train_ratings = []
test_ratings = []
for user_id in ratings['UserID'].unique():
    user_r = ratings[ratings['UserID'] == user_id]
    if len(user_r) < 5:
        continue
    train, test = train_test_split(user_r, test_size=0.2, random_state=42)
    train_ratings.append(train)
    test_ratings.append(test)
train_ratings = pd.concat(train_ratings)
test_ratings = pd.concat(test_ratings)

# Prepare test ratings by user for ranking metrics
test_ratings_by_user = test_ratings.groupby('UserID')['MovieID'].apply(list)
train_movies_by_user = train_ratings.groupby('UserID')['MovieID'].apply(set)

# Compute metrics for each model
metrics = {m: {'Precision@10': [], 'Recall@10': [], 'MAP@10': [], 'NDCG@10': []} for m in MODELS}
for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        continue
    model, trainset, user_items = joblib.load(model_path)
    for user_id, relevant_movies in test_ratings_by_user.items():
        if user_id not in user_items:
            continue
        seen = train_movies_by_user.get(user_id, set())
        all_inner_ids = set(trainset.all_items())
        all_movie_ids = set(trainset.to_raw_iid(iid) for iid in all_inner_ids)
        candidates = all_movie_ids - seen
        recs = []
        for iid in candidates:
            try:
                recs.append((iid, model.predict(user_id, iid).est))
            except Exception:
                continue
        recs = sorted(recs, key=lambda x: x[1], reverse=True)
        rec_ids = [iid for iid, _ in recs[:10]]
        metrics[model_name]['Precision@10'].append(precision_at_k(relevant_movies, rec_ids, k=10))
        metrics[model_name]['Recall@10'].append(recall_at_k(relevant_movies, rec_ids, k=10))
        metrics[model_name]['MAP@10'].append(map_at_k(relevant_movies, rec_ids, k=10))
        metrics[model_name]['NDCG@10'].append(ndcg_at_k(relevant_movies, rec_ids, k=10))

# Select best model by highest mean MAP@10
best_model_name = max(metrics, key=lambda m: np.mean(metrics[m]['MAP@10']) if metrics[m]['MAP@10'] else -1)
best_model_path = MODELS[best_model_name]
print(f"Best model: {best_model_name} (MAP@10={np.mean(metrics[best_model_name]['MAP@10']):.4f})")
model, trainset, user_items = joblib.load(best_model_path)

# Plot bar chart of best model's metrics
metric_names = ['Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10']
means = [np.mean(metrics[best_model_name][m]) for m in metric_names]
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(metric_names, means, color=sns.color_palette()[4])
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11)
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title(f'{best_model_name} Performance Across Metrics')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'{best_model_name}_metrics_bar.png'))
plt.show()

# Latent factors visualization
if hasattr(model, 'pu') and hasattr(model, 'qi'):
    user_factors = model.pu
    item_factors = model.qi
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(user_factors.flatten(), bins=30, kde=True)
    plt.title(f'{best_model_name} User Latent Factors')
    plt.subplot(1,2,2)
    sns.histplot(item_factors.flatten(), bins=30, kde=True)
    plt.title(f'{best_model_name} Item Latent Factors')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{best_model_name}_latent_factors.png'))
    plt.show()
# Predicted ratings histogram
all_users = list(trainset.all_users())
all_items = list(trainset.all_items())
user_ids = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)
item_ids = np.random.choice(all_items, size=min(50, len(all_items)), replace=False)
preds = []
for u in user_ids:
    for i in item_ids:
        uid = trainset.to_raw_uid(u)
        iid = trainset.to_raw_iid(i)
        try:
            pred = model.predict(uid, iid).est
            preds.append(pred)
        except Exception:
            continue
plt.figure(figsize=(6,4))
sns.histplot(preds, bins=30, kde=True)
plt.title(f'{best_model_name} Predicted Ratings (sample)')
plt.xlabel('Predicted Rating')
plt.savefig(os.path.join(PLOTS_DIR, f'{best_model_name}_predicted_ratings_hist.png'))
plt.show()
# Top-10 recommendations for 3 random users
print(f"Top-10 recommendations for 3 random users (by MovieID):")
for uid in np.random.choice(all_users, size=min(3, len(all_users)), replace=False):
    raw_uid = trainset.to_raw_uid(uid)
    seen = user_items.get(raw_uid, set())
    all_movie_ids = set(trainset.to_raw_iid(i) for i in all_items)
    candidates = all_movie_ids - seen
    recs = []
    for iid in candidates:
        try:
            recs.append((iid, model.predict(raw_uid, iid).est))
        except Exception:
            continue
    recs = sorted(recs, key=lambda x: x[1], reverse=True)[:10]
    rec_titles = [list(movies[movies['MovieID'] == int(mid)]['Title'])[0] if not movies[movies['MovieID'] == int(mid)].empty else str(mid) for mid, _ in recs]
    print(f"User {raw_uid}: {rec_titles}") 