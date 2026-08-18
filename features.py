import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "danceability","energy","valence","acousticness",
    "instrumentalness","speechiness","liveness","tempo",
    "loudness","duration_ms"
]

def load_tracks(path="data/sample_tracks.csv"):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required audio features: {missing}")
    return df.dropna(subset=FEATURES).reset_index(drop=True)

def make_matrix(df):
    X = df[FEATURES].copy()
    X["tempo"] = X["tempo"] / 200.0
    X["loudness"] = (X["loudness"] + 20.0) / 20.0
    X["duration_ms"] = X["duration_ms"] / 300000.0
    X = X.clip(0, 1)
    return StandardScaler().fit_transform(X)
