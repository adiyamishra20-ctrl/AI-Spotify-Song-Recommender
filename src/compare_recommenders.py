import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_recommender import load_data, recommend


def print_results(label, results):
    print(f"\n{label}")
    print("-" * len(label))

    if results.empty:
        print("No matching recommendations.")
        return

    for number, song in enumerate(results.to_dict("records"), start=1):
        print(
            f"{number}. {song['title']} — {song['artist']}"
            f"\n   Similarity: {song['similarity']:.3f}"
            f"\n   Moods: {', '.join(song['moods'])}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--mood")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    songs, raw_features = load_data()

    embeddings = pd.read_csv(
        root / "data" / "processed" / "song_embeddings.csv"
    )

    if (
        embeddings["row_id"].duplicated().any()
        or not songs["row_id"].equals(embeddings["row_id"])
    ):
        raise ValueError("Embedding rows do not align with song rows.")

    embedding_columns = [f"embedding_{i}" for i in range(16)]
    matrix = embeddings[embedding_columns].to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain invalid values.")

    if (np.linalg.norm(matrix, axis=1) < 1e-12).any():
        raise ValueError("A song has a zero-length embedding.")

    options = {
        "title": args.title,
        "artist": args.artist,
        "mood": args.mood,
        "top_k": args.top_k,
    }

    try:
        baseline = recommend(songs, raw_features, **options)
        autoencoder = recommend(songs, matrix, **options)
    except ValueError as error:
        parser.error(str(error))

    print(f"Reference: {args.title} — {args.artist}")
    print(f"Mood filter: {args.mood or 'none'}")

    print_results("Raw-feature baseline", baseline)
    print_results("Autoencoder embeddings", autoencoder)

    overlap = set(baseline["row_id"]) & set(autoencoder["row_id"])
    print(f"\nShared recommendations: {len(overlap)}")

    # Save both lists for inspection.
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    comparison = pd.concat(
        [
            baseline.assign(method="raw_features"),
            autoencoder.assign(method="autoencoder"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(
        reports / "recommendation_comparison.csv", index=False
    )

    print("Saved reports/recommendation_comparison.csv")


if __name__ == "__main__":
    main()