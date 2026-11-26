import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean data
df = pd.read_csv("data/sptify_tracks.csv")
df = df.drop(columns=["Unnamed: 0"]).dropna().drop_duplicates()
df["explicit"] = df["explicit"].astype(int)

# 2. Define target variable ("Hit" = popularity >= 70)
df["is_hit"] = (df["popularity"] >= 70).astype(int)

features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "explicit"
]

X = df[features]
y = df["is_hit"]

# 3. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features (essential for distance-based models like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train KNN model (checking the 5 nearest songs)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# 6. Evaluate performance
print("Evaluating KNN (this may take 10–15 seconds)...")
y_pred = knn.predict(X_test_scaled)
print("KNN Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))