
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sptify_tracks.csv")

print("Original Dataset:", df.shape)

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Remove rows with missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert explicit into numerical values
df["explicit"] = df["explicit"].astype(int)

# Select audio features
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

X = df[features]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPreprocessing completed!")
print("Clean Dataset:", df.shape)
print("Features Shape:", X_scaled.shape)
print("Missing Values:", df.isnull().sum().sum())

