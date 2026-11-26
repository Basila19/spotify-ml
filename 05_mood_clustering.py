import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Load and clean data
df = pd.read_csv("data/sptify_tracks.csv")
df = df.drop(columns=["Unnamed: 0"]).dropna().drop_duplicates()

# 2. Select mood-defining audio features
mood_features = [
    "danceability", "energy", "loudness",
    "acousticness", "instrumentalness", "valence", "tempo"
]

X = df[mood_features]

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reduce dimensions with PCA (keeping 90% of variance)
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]} components.")

# 5. Group songs into 4 Mood Clusters using K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["mood_cluster"] = kmeans.fit_predict(X_pca)

# 6. Inspect the average audio profile of each mood cluster
print("\nCluster Profiles (Average Audio Features):")
cluster_summary = df.groupby("mood_cluster")[mood_features].mean().round(2)
print(cluster_summary)

print("\nNumber of songs in each mood cluster:")
print(df["mood_cluster"].value_counts())