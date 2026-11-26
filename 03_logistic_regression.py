import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and clean data
df = pd.read_csv("data/sptify_tracks.csv")
df = df.drop(columns=["Unnamed: 0"]).dropna().drop_duplicates()
df["explicit"] = df["explicit"].astype(int)

# 2. Define what counts as a "Hit" (popularity >= 70 -> 1, else -> 0)
df["is_hit"] = (df["popularity"] >= 70).astype(int)

features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "explicit"
]

X = df[features]
y = df["is_hit"]

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features (fit only on training data to prevent data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# 6. Evaluate performance
y_pred = model.predict(X_test_scaled)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))