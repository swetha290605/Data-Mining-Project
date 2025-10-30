import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load cleaned dataset
df = pd.read_csv("ehr_dataset_cleaned.csv")

# Step 2: Keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Step 3: Standardize (normalize) the data
scaler = StandardScaler()
X = scaler.fit_transform(numeric_df)

# Step 4: Reduce to 2 dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 5: Initialize dictionary for results
results = {}

# Step 6: Make sure 'plots' folder exists
os.makedirs("plots", exist_ok=True)

# Helper function to visualize and save clusters
def plot_clusters(X_2d, labels, title, filename):
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette="Set2", s=50)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", loc="best")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved plot: {filename}")

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)
results["KMeans"] = {
    "Silhouette": silhouette_score(X, labels_kmeans),
    "Davies-Bouldin": davies_bouldin_score(X, labels_kmeans),
    "Calinski-Harabasz": calinski_harabasz_score(X, labels_kmeans)
}
plot_clusters(X_pca, labels_kmeans, "K-Means Clustering", "plots/kmeans_plot.png")

# --- Agglomerative Clustering ---
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X)
results["Agglomerative"] = {
    "Silhouette": silhouette_score(X, labels_agg),
    "Davies-Bouldin": davies_bouldin_score(X, labels_agg),
    "Calinski-Harabasz": calinski_harabasz_score(X, labels_agg)
}
plot_clusters(X_pca, labels_agg, "Agglomerative Clustering", "plots/agg_plot.png")

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    results["DBSCAN"] = {
        "Silhouette": silhouette_score(X, labels_dbscan),
        "Davies-Bouldin": davies_bouldin_score(X, labels_dbscan),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels_dbscan)
    }
else:
    results["DBSCAN"] = {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}

plot_clusters(X_pca, labels_dbscan, "DBSCAN Clustering", "plots/dbscan_plot.png")

# Step 7: Print results
print("\n=== Clustering Evaluation Metrics ===")
for method, metrics in results.items():
    print(f"\n{method}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score}")

# Step 8: Determine best method (by highest Silhouette score)
best_method = max(results.items(), key=lambda x: x[1]["Silhouette"] or -1)[0]
print(f"\n Best clustering method (by Silhouette score): {best_method}")
