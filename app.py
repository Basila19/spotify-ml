import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Spotify Hit & Mood Engine", page_icon="🎧", layout="wide")

# Load saved models
@st.cache_resource
def load_models():
    return joblib.load("spotify_engine_models.pkl")

artifacts = load_models()
hit_model = artifacts["hit_model"]
hit_features = artifacts["hit_features"]
mood_scaler = artifacts["mood_scaler"]
pca = artifacts["pca"]
kmeans = artifacts["kmeans"]
mood_features = artifacts["mood_features"]

MOOD_NAMES = {
    0: "Happy & Energetic",
    1: "Moody & Energetic",
    2: "Chill & Acoustic",
    3: "Ambient & Instrumental"
}

MOOD_DESCRIPTIONS = {
    0: "Happy & Energetic (High danceability)",
    1: "Moody & Energetic (Fast & intense)",
    2: "Chill & Acoustic (Laid-back vibe)",
    3: "Ambient & Instrumental (Calm & peaceful)"
}

COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

# Helper to detect Indian/regional language tags from track/album/genre text
def detect_language(row):
    text = f"{row['track_name']} {row['album_name']} {row['track_genre']}".lower()
    if "telugu" in text:
        return "Telugu"
    elif "tamil" in text:
        return "Tamil"
    elif "hindi" in text or "bollywood" in text:
        return "Hindi"
    elif "malayalam" in text:
        return "Malayalam"
    elif "punjabi" in text:
        return "Punjabi"
    elif "kannada" in text:
        return "Kannada"
    elif "spanish" in text or "latin" in text:
        return "Spanish"
    elif "korean" in text or "k-pop" in text:
        return "Korean"
    return "Other / not specified"

# Precompute predictions & mood clusters for the whole dataset once
@st.cache_data
def prepare_catalog(_hit_model, hit_features, _mood_scaler, _pca, _kmeans, mood_features):
    df = pd.read_csv("data/sptify_tracks.csv").drop(columns=["Unnamed: 0"]).dropna().drop_duplicates()
    df["explicit"] = df["explicit"].astype(int)

    # Predict hit probability for all tracks
    probs = _hit_model.predict_proba(df[hit_features])[:, 1]
    df["Hit score (%)"] = np.round(probs * 100).astype(int)

    # Predict mood cluster & PCA coords for all tracks
    X_scaled = _mood_scaler.transform(df[mood_features])
    X_pca = _pca.transform(X_scaled)
    df["cluster_id"] = _kmeans.predict(X_pca)
    df["Mood"] = df["cluster_id"].map(MOOD_DESCRIPTIONS)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    # Language & Spotify URL
    df["Language"] = df.apply(detect_language, axis=1)
    df["Spotify"] = "https://open.spotify.com/track/" + df["track_id"].astype(str)
    return df

catalog_df = prepare_catalog(hit_model, hit_features, mood_scaler, pca, kmeans, mood_features)

# --- SIDEBAR SLIDERS (Used in Tab 1) ---
st.sidebar.header("🎛️ Track Audio Features")
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.55)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.75)
valence = st.sidebar.slider("Valence (Positivity)", 0.0, 1.0, 0.35)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.15)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.10)
loudness = st.sidebar.slider("Loudness (dB)", -40.0, 0.0, -6.0)
tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 210.0, 130.0)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.08)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.15)
duration_sec = st.sidebar.slider("Duration (seconds)", 60, 420, 210)
explicit = st.sidebar.checkbox("Explicit Lyrics", value=False)

# --- HEADER & TABS ---
st.title("🎧 Spotify Hit & Mood Engine")
st.caption("Hit = popularity ≥ 70. Classifier: Tuned Boosting (decision threshold 0.50).")

tab1, tab2, tab3 = st.tabs(["Predict a song", "Model results", "Discover songs"])

# ==========================================
# TAB 1: PREDICT A SONG
# ==========================================
with tab1:
    st.subheader("Prediction")

    input_data = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_sec * 1000,
        "explicit": int(explicit)
    }])

    hit_prob = float(hit_model.predict_proba(input_data[hit_features])[0][1])
    is_hit = int(hit_model.predict(input_data[hit_features])[0])

    mood_scaled = mood_scaler.transform(input_data[mood_features])
    mood_pca = pca.transform(mood_scaled)
    cluster_id = int(kmeans.predict(mood_pca)[0])

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Hit probability", f"{int(round(hit_prob * 100))}%")
    with c2:
        st.metric("Mood cluster", MOOD_NAMES[cluster_id])

    st.progress(hit_prob)

    if is_hit == 1:
        st.success("🔥 Likely to be a Hit!")
    else:
        st.info("Probably not a hit")

    # 2D PCA Cluster Plot
    sample_plot = catalog_df.sample(n=4000, random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid in range(4):
        sub = sample_plot[sample_plot["cluster_id"] == cid]
        ax.scatter(
            sub["PC1"], sub["PC2"],
            c=COLORS[cid], label=MOOD_DESCRIPTIONS[cid],
            alpha=0.35, s=18, edgecolors="none"
        )

    ax.scatter(
        mood_pca[0, 0], mood_pca[0, 1],
        c="black", marker="*", s=350, zorder=10, label="Your Track"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# TAB 2: MODEL RESULTS
# ==========================================
with tab2:
    st.subheader("Model Performance Comparison")
    st.write("Comparison of classifiers trained on 113,549 Spotify tracks:")

    results_df = pd.DataFrame({
        "Model": ["Logistic Regression (Baseline)", "K-Nearest Neighbors (KNN)", "HistGradientBoosting (Tuned)"],
        "Role": ["Linear Baseline", "Distance-Based Classifier", "Final Production Model"],
        "Accuracy (%)": [71.4, 93.8, 75.0],
        "Notes": [
            "Balanced class weights; high recall for hits",
            "Sensitive to local feature density",
            "Best probability calibration & fast inference"
        ]
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.subheader("Mood Clustering Summary (PCA + K-Means, k=4)")
    cluster_profile = catalog_df.groupby("Mood")[mood_features].mean().round(2).reset_index()
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: DISCOVER SONGS
# ==========================================
with tab3:
    st.subheader("Discover songs")

    f1, f2, f3 = st.columns(3)
    with f1:
        mood_options = ["All"] + list(MOOD_DESCRIPTIONS.values())
        selected_mood = st.selectbox("Mood category", mood_options)
    with f2:
        genre_options = ["All"] + sorted(catalog_df["track_genre"].unique().tolist())
        selected_genre = st.selectbox("Genre category", genre_options)
    with f3:
        lang_options = ["Any"] + sorted(catalog_df["Language"].unique().tolist())
        selected_lang = st.selectbox("Language (optional)", lang_options)

    s1, s2, s3 = st.columns([2, 1.2, 1])
    with s1:
        search_query = st.text_input("Search song or artist (optional)", "")
    with s2:
        sort_by = st.radio("Sort by", ["Highest hit score", "Most popular"], horizontal=True)
    with s3:
        limit_rows = st.number_input("How many", min_value=5, max_value=200, value=25, step=5)

    # Apply filters
    filtered = catalog_df.copy()
    if selected_mood != "All":
        filtered = filtered[filtered["Mood"] == selected_mood]
    if selected_genre != "All":
        filtered = filtered[filtered["track_genre"] == selected_genre]
    if selected_lang != "Any":
        filtered = filtered[filtered["Language"] == selected_lang]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[
            filtered["track_name"].str.lower().str.contains(q, na=False) |
            filtered["artists"].str.lower().str.contains(q, na=False)
        ]

    # Sort & deduplicate display rows
    sort_col = "Hit score (%)" if sort_by == "Highest hit score" else "popularity"
    filtered = (
        filtered.sort_values(by=[sort_col, "popularity"], ascending=False)
        .drop_duplicates(subset=["track_name", "artists"])
        .head(int(limit_rows))
    )

    display_df = filtered[[
        "track_name", "artists", "track_genre", "Mood", "Language", "popularity", "Hit score (%)", "Spotify"
    ]].rename(columns={
        "track_name": "Song",
        "artists": "Artist",
        "track_genre": "Genre",
        "popularity": "Popularity"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spotify": st.column_config.LinkColumn(
                "Spotify",
                display_text="Open ▶"
            )
        }
    )