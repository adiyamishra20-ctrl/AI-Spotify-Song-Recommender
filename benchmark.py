from pathlib import Path
import sys, numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features import load_tracks, make_matrix
from src.model import train_autoencoder

df = load_tracks(ROOT / "data" / "sample_tracks.csv")
X = make_matrix(df)
_, Z = train_autoencoder(X)

def mood_hit_rate(M):
    sims = cosine_similarity(M)
    np.fill_diagonal(sims, -np.inf)
    nn = sims.argmax(axis=1)
    return (df["mood"].values[nn] == df["mood"].values).mean()

raw = mood_hit_rate(X)
latent = mood_hit_rate(Z)
improvement = ((latent-raw)/raw*100) if raw else 0

print(f"Raw-feature mood hit rate: {raw:.3f}")
print(f"Autoencoder mood hit rate: {latent:.3f}")
print(f"Relative improvement: {improvement:.1f}%")
print("Use this measured value—not a preset claim—in your CV.")
