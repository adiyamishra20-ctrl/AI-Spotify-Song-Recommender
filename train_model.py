from pathlib import Path
import sys, json, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import load_tracks, make_matrix
from src.model import train_autoencoder

df = load_tracks(ROOT / "data" / "sample_tracks.csv")
X = make_matrix(df)
model, Z = train_autoencoder(X)

OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
torch.save(model.state_dict(), OUT / "autoencoder.pt")
np.save(OUT / "embeddings.npy", Z)
(OUT / "metadata.json").write_text(json.dumps({
    "rows": len(df), "latent_dimensions": Z.shape[1],
    "dataset": "synthetic demo dataset"
}, indent=2))
print(f"Saved embeddings for {len(df)} tracks.")
