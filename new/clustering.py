import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Step 1: Loading cleaned dataset
df = pd.read_csv("ehr_dataset_cleaned.csv")  # make sure this file is in the same folder

# Step 2: taking only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Step 3: Standardize (normalize) the data
scaler = StandardScaler()
X = scaler.fit_transform(numeric_df)

# Step 4: Initializing dictionary for results
results = {}

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)
results["KMeans"] = {
    "Silhouette": silhouette_score(X, labels_kmeans),
    "Davies-Bouldin": davies_bouldin_score(X, labels_kmeans),
    "Calinski-Harabasz": calinski_harabasz_score(X, labels_kmeans)
}

# --- Agglomerative Clustering ---
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X)
results["Agglomerative"] = {
    "Silhouette": silhouette_score(X, labels_agg),
    "Davies-Bouldin": davies_bouldin_score(X, labels_agg),
    "Calinski-Harabasz": calinski_harabasz_score(X, labels_agg)
}

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Handle case where all points are noise or only 1 cluster
if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    results["DBSCAN"] = {
        "Silhouette": silhouette_score(X, labels_dbscan),
        "Davies-Bouldin": davies_bouldin_score(X, labels_dbscan),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels_dbscan)
    }
else:
    results["DBSCAN"] = {"Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}

# Step 5: Printing results
print("\n=== Clustering Evaluation Metrics ===")
for method, metrics in results.items():
    print(f"\n{method}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score}")

# Step 6: Determining best method (based on highest Silhouette score)
best_method = max(results.items(), key=lambda x: x[1]["Silhouette"] or -1)[0]
print(f"\nBest clustering method (by Silhouette score): {best_method}")
