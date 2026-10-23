import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data" / "processed"

    songs = pd.read_csv(folder / "baseline_songs.csv")
    features = pd.read_csv(folder / "baseline_features.csv")

    if (
        songs["row_id"].duplicated().any()
        or features["row_id"].duplicated().any()
        or not songs["row_id"].equals(features["row_id"])
    ):
        raise ValueError("Song rows and feature rows are not aligned.")

    matrix = features.drop(columns="row_id").to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError("Features contain missing or invalid numbers.")

    if (np.linalg.norm(matrix, axis=1) == 0).any():
        raise ValueError("A song has no usable features.")

    songs["genres"] = songs["genres"].map(json.loads)
    songs["moods"] = songs["moods"].map(json.loads)

    return songs, matrix


def recommend(songs, matrix, title, artist, mood=None, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    matches = (
        songs["title"].str.casefold().eq(title.strip().casefold())
        & songs["artist"].str.casefold().eq(artist.strip().casefold())
    )
    positions = np.flatnonzero(matches.to_numpy())

    if len(positions) != 1:
        raise ValueError(
            f"Expected one reference song, found {len(positions)}. "
            "Check the title and artist in baseline_songs.csv."
        )

    reference_position = int(positions[0])
    scores = cosine_similarity(
        matrix[reference_position:reference_position + 1],
        matrix,
    )[0]

    candidates = songs.copy()
    candidates["similarity"] = scores

    # Never recommend the reference row itself.
    candidates = candidates[
        candidates["row_id"] != songs.iloc[reference_position]["row_id"]
    ]

    if mood:
        mood = mood.strip().casefold()
        available = {value for values in songs["moods"] for value in values}

        if mood not in available:
            raise ValueError(
                f"Unknown mood '{mood}'. Choose from: "
                + ", ".join(sorted(available))
            )

        candidates = candidates[
            candidates["moods"].map(lambda values: mood in values)
        ]

    # Deterministic ordering when songs have equal similarity.
    return candidates.sort_values(
        ["similarity", "title", "artist", "row_id"],
        ascending=[False, True, True, True],
    ).head(top_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--mood")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    songs, matrix = load_data()

    try:
        results = recommend(
            songs,
            matrix,
            title=args.title,
            artist=args.artist,
            mood=args.mood,
            top_k=args.top_k,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"\nReference: {args.title} — {args.artist}")
    print(f"Mood filter: {args.mood or 'none'}")

    if results.empty:
        print("No recommendations match this filter.")
        return

    for number, song in enumerate(results.to_dict("records"), start=1):
        print(
            f"\n{number}. {song['title']} — {song['artist']}"
            f"\n   Similarity: {song['similarity']:.3f}"
            f"\n   Genres: {', '.join(song['genres'])}"
            f"\n   Moods: {', '.join(song['moods'])}"
        )


if __name__ == "__main__":
    main()