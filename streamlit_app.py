from pathlib import Path
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features import make_matrix
from src.model import train_autoencoder

st.set_page_config(page_title="AI Spotify Song Picker", page_icon="🎧", layout="wide")
st.title("🎧 AI Spotify Song Picker")
st.caption("Mood-aware content recommendation using learned song embeddings")

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/sample_tracks.csv")

@st.cache_resource
def build_embeddings(df):
    X = make_matrix(df)
    _, Z = train_autoencoder(X)
    return Z

df = load_data()
Z = build_embeddings(df)

mood = st.selectbox("Choose a mood", sorted(df["mood"].unique()))
n = st.slider("Recommendations", 3, 10, 5)

mask = df["mood"].eq(mood).values
query = Z[mask].mean(axis=0, keepdims=True)
sims = cosine_similarity(Z, query).ravel()
order = np.argsort(-sims)
idx = [i for i in order if not mask[i]][:n]

st.subheader(f"Recommendations for {mood.title()}")
result = df.iloc[idx][["track_name","artists","genre","mood"]].copy()
result["similarity"] = np.round(sims[idx], 3)
st.dataframe(result, use_container_width=True, hide_index=True)

def explain(row, score):
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=token)
            prompt = (
                f"Explain in one concise sentence why '{row.track_name}' by "
                f"{row.artists} fits a {mood} mood. Mention its genre. "
                "Do not invent lyrics or facts."
            )
            out = client.chat_completion(
                model=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                messages=[{"role":"user","content":prompt}],
                max_tokens=60, temperature=0.3
            )
            return out.choices[0].message.content.strip()
        except Exception:
            pass
    return (
        f"Recommended from the learned audio-feature embedding for a "
        f"{mood} context (similarity {score:.2f})."
    )

for i in idx[:3]:
    st.markdown(f"**{df.iloc[i]['track_name']} — {df.iloc[i]['artists']}**")
    st.write(explain(df.iloc[i], sims[i]))
