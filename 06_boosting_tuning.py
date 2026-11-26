import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean data
df = pd.read_csv("data/sptify_tracks.csv")
df = df.drop(columns=["Unnamed: 0"]).dropna().drop_duplicates()
df["explicit"] = df["explicit"].astype(int)
df["is_hit"] = (df["popularity"] >= 70).astype(int)

# --- PART A: HIT PREDICTION MODEL (BOOSTING) ---
hit_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "explicit"
]

X = df[hit_features]
y = df["is_hit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tuned Gradient Boosting model
hit_model = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.08,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)
hit_model.fit(X_train, y_train)

y_pred = hit_model.predict(X_test)
print("Boosting Hit Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- PART B: MOOD CLUSTERING PIPELINE ---
mood_features = [
    "danceability", "energy", "loudness",
    "acousticness", "instrumentalness", "valence", "tempo"
]

mood_scaler = StandardScaler()
X_mood_scaled = mood_scaler.fit_transform(df[mood_features])

pca = PCA(n_components=0.90, random_state=42)
X_mood_pca = pca.fit_transform(X_mood_scaled)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["mood_cluster"] = kmeans.fit_predict(X_mood_pca)

# Inspect cluster averages so we can name the moods accurately
print("\nMood Cluster Averages:")
print(df.groupby("mood_cluster")[mood_features].mean().round(2))

# --- PART C: SAVE MODELS FOR STREAMLIT APP ---
artifacts = {
    "hit_model": hit_model,
    "hit_features": hit_features,
    "mood_scaler": mood_scaler,
    "pca": pca,
    "kmeans": kmeans,
    "mood_features": mood_features
}

joblib.dump(artifacts, "spotify_engine_models.pkl")
print("\nSaved all trained models to 'spotify_engine_models.pkl'!")