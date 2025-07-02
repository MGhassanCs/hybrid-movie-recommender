import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PLOTS_DIR = 'plots'
model_files = {
    'Content-Based': 'content_metrics.json',
    'Hybrid': 'hybrid_metrics.json',
    'SVD': 'svd_metrics.json',
    'SVD++': 'svdpp_metrics.json',
    'NMF': 'nmf_metrics.json',
}

# Load metrics for each model
all_metrics = {}
for model, filename in model_files.items():
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            all_metrics[model] = json.load(f)
    else:
        print(f"Warning: {filename} not found, skipping {model}.")

metric_names = ['Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10']
model_names = list(all_metrics.keys())
bar_width = 0.15
x = np.arange(len(metric_names))

fig, ax = plt.subplots(figsize=(10, 6))
for i, model in enumerate(model_names):
    values = [all_metrics[model][m] for m in metric_names]
    positions = x + (i - len(model_names)/2) * bar_width
    bars = ax.bar(positions, values, bar_width, label=model)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('All Model Comparison Across Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend(title='Model')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'all_model_metrics_bar.png'))
plt.show() 