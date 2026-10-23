import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline_recommender import load_data
from src.explanations import explain_recommendation


st.set_page_config(
    page_title="Song Compass",
    page_icon="🎵",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent


def identity(title, artist):
    return (
        " ".join(str(title).casefold().split()),
        " ".join(str(artist).casefold().split()),
    )


@st.cache_resource
def load_models():
    songs, raw = load_data()
    frame = pd.read_csv(
        ROOT / "data" / "processed" / "song_embeddings.csv"
    )

    if (
        frame["row_id"].duplicated().any()
        or not songs["row_id"].equals(frame["row_id"])
    ):
        raise ValueError("Embedding rows do not align with songs.")

    embeddings = frame[
        [f"embedding_{i}" for i in range(16)]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(embeddings).all()
        or (np.linalg.norm(embeddings, axis=1) < 1e-12).any()
    ):
        raise ValueError("Invalid embeddings.")

    return songs, raw, embeddings


def load_reviews():
    path = ROOT / "reports" / "reference_mood_review.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    reviews = {}

    for row in frame.to_dict("records"):
        key = (*identity(row["title"], row["artist"]), row["mood"])
        judgment = row["judgment"]

        if key in reviews and reviews[key] != judgment:
            raise ValueError(f"Conflicting reviews for {row['title']}.")

        reviews[key] = judgment

    return reviews


def mood_status(song, mood, reviews):
    key = (*identity(song["title"], song["artist"]), mood)
    judgment = reviews.get(key, "Unreviewed")

    if judgment == "Supported":
        return "Reviewed match"
    if judgment in {"Unsupported", "Uncertain"}:
        return "Excluded"
    if mood in song["moods"]:
        return "Dataset-only match"
    return "Excluded"


def effective_moods(song, reviews):
    # Prevent explanations from repeating rejected or uncertain tags.
    return sorted(
        mood for mood in song["moods"]
        if reviews.get(
            (*identity(song["title"], song["artist"]), mood),
            "Unreviewed",
        ) not in {"Unsupported", "Uncertain"}
    )


st.title("🎵 Song Compass")
st.write("How do you want your music to feel?")

try:
    songs, raw_features, embeddings = load_models()
    reviews = load_reviews()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

st.caption(
    "Personal reviews control mood eligibility. Dataset-only annotations "
    "are unverified listener tags."
)

moods = sorted({mood for values in songs["moods"] for mood in values})
genres = sorted({genre for values in songs["genres"] for genre in values})

with st.form("mood_search"):
    mood = st.selectbox("Choose your mood", moods)
    genre = st.selectbox("Optional genre", ["Any genre"] + genres)

    reviewed_only = st.checkbox(
        "Only show moods I marked Supported",
        value=True,
    )

    reference_position = st.selectbox(
        "Optional reference song",
        options=[None] + list(range(len(songs))),
        format_func=lambda position: (
            "No reference song"
            if position is None
            else (
                f"{songs.iloc[position]['title']} — "
                f"{songs.iloc[position]['artist']}"
            )
        ),
    )

    method = st.radio(
        "Similarity method — used only with a reference song",
        ["Raw-feature baseline", "Autoencoder"],
        index=0,
        horizontal=True,
    )

    st.caption(
        "Raw-feature ranking is the default based on held-out evaluation. "
        "Autoencoder ranking is experimental."
    )
    count = st.slider("Number of songs", 3, 15, 5)
    submitted = st.form_submit_button("Find songs", type="primary")

if submitted:
    st.session_state["mood_request"] = {
        "mood": mood,
        "genre": genre,
        "reviewed_only": reviewed_only,
        "reference_position": reference_position,
        "method": method,
        "count": count,
    }

if "mood_explanations" not in st.session_state:
    st.session_state["mood_explanations"] = {}

if "mood_request" in st.session_state:
    request = st.session_state["mood_request"]
    selected_mood = request["mood"]

    candidates = songs.copy()
    candidates["mood_status"] = [
        mood_status(song, selected_mood, reviews)
        for song in candidates.to_dict("records")
    ]

    allowed = (
        ["Reviewed match"]
        if request["reviewed_only"]
        else ["Reviewed match", "Dataset-only match"]
    )
    candidates = candidates[candidates["mood_status"].isin(allowed)]

    if request["genre"] != "Any genre":
        genre_mask = candidates["genres"].map(
            lambda values: request["genre"] in values
        ).astype(bool)

        candidates = candidates.loc[genre_mask].copy()

    if candidates.empty:
        st.info(
            "No songs match these settings. Try another genre or mood, "
            "or uncheck reviewed-only to allow dataset-only matches."
        )
        st.stop()

    position = request["reference_position"]
    reference = None

    if position is not None:
        reference = songs.iloc[position]
        matrix = (
            embeddings
            if request["method"] == "Autoencoder"
            else raw_features
        )
        scores = cosine_similarity(
            matrix[position:position + 1], matrix
        )[0]
        candidates["similarity"] = scores[candidates.index.to_numpy()]
        candidates = candidates[
            candidates["row_id"] != reference["row_id"]
        ]

    candidates["review_priority"] = (
        candidates["mood_status"].eq("Dataset-only match").astype(int)
    )

    if reference is not None:
        candidates = candidates.sort_values(
            ["review_priority", "similarity", "title", "artist"],
            ascending=[True, False, True, True],
        )
    else:
        # No invented mood score: reviewed songs first, then alphabetical.
        candidates = candidates.sort_values(
            ["review_priority", "title", "artist"]
        )

    results = candidates.head(request["count"])

    st.subheader(f"Songs for: {selected_mood}")
    st.caption(
        f"{len(candidates)} eligible songs; showing {len(results)}. "
        + (
            "Reviewed matches first, then reference-song similarity."
            if reference is not None
            else "Reviewed matches first, then alphabetical order."
        )
    )

    if results.empty:
        st.info(
            "No songs meet these settings. Review more songs for this mood, "
            "change the genre, or explicitly allow dataset-only matches."
        )

    for song in results.to_dict("records"):
        with st.container(border=True):
            st.subheader(f"{song['title']} — {song['artist']}")
            st.write(f"**{song['mood_status']} · {selected_mood}**")
            st.caption("Genres: " + ", ".join(song["genres"]))

            if song["mood_status"] == "Reviewed match":
                st.write(f"You marked this song as fitting {selected_mood}.")
            else:
                st.write(
                    f"The dataset includes the tag {selected_mood}; "
                    "you have not verified it."
                )

            st.link_button(
                "Search on Spotify",
                "https://open.spotify.com/search/"
                + quote(f"{song['title']} {song['artist']}", safe=""),
            )

            # Existing Llama function explains a reference–candidate pair.
            if reference is not None:
                reference_data = {
                    "title": reference["title"],
                    "genres": list(reference["genres"]),
                    "moods": effective_moods(reference, reviews),
                }
                candidate_data = {
                    "title": song["title"],
                    "genres": list(song["genres"]),
                    "moods": effective_moods(song, reviews),
                }

                cache_key = json.dumps(
                    [reference_data, candidate_data, selected_mood],
                    sort_keys=True,
                )
                explanation = st.session_state["mood_explanations"].get(
                    cache_key
                )

                if st.button(
                    "Explain reference-song match",
                    key=f"explain_{song['row_id']}",
                ):
                    with st.spinner("Preparing explanation…"):
                        explanation = explain_recommendation(
                            reference_data, candidate_data, selected_mood
                        )
                    if explanation["source"] == "Pre-generated Llama 3.2 explanation":
                        st.session_state["mood_explanations"][cache_key] = (
                            explanation
                        )

                if explanation:
                    st.write(explanation["text"])
                    st.caption(
                        f"Source: {explanation['source']}. "
                        "Annotation-based explanation, not an audio analysis."
                    )

st.caption(
    "The existing embeddings still contain the original dataset tags. "
    "Your reviews override eligibility and explanation inputs; "
    "the model has not yet been retrained on corrected labels."
)