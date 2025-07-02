import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = 'plots'
metrics_path = os.path.join(PLOTS_DIR, 'content_metrics.json')

with open(metrics_path, 'r') as f:
    metrics = json.load(f)

metric_names = list(metrics.keys())
means = [metrics[m] for m in metric_names]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(metric_names, means, color=sns.color_palette()[0])
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11)
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Content-Based Model Performance Across Metrics')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'content_model_metrics_bar.png'))
plt.show() 