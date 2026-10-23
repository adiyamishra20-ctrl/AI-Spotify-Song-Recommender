import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_recommender import load_data, recommend


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "reports" / "evaluation_pilot"

    # Preserve existing evaluation files and ratings.
    if folder.exists():
        raise FileExistsError(
            "evaluation_pilot already exists. Keep it intact; "
            "do not regenerate an evaluation after seeing its results."
        )

    songs, raw_features = load_data()

    embeddings = pd.read_csv(
        root / "data" / "processed" / "song_embeddings.csv"
    )
    splits = pd.read_csv(root / "reports" / "autoencoder_split.csv")

    if (
        embeddings["row_id"].duplicated().any()
        or not songs["row_id"].equals(embeddings["row_id"])
    ):
        raise ValueError("Embedding IDs do not align with songs.")

    if (
        splits["row_id"].duplicated().any()
        or set(splits["row_id"]) != set(songs["row_id"])
    ):
        raise ValueError("Split IDs do not align with songs.")

    matrix = embeddings[
        [f"embedding_{i}" for i in range(16)]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(matrix).all()
        or (np.linalg.norm(matrix, axis=1) < 1e-12).any()
    ):
        raise ValueError("Invalid embeddings.")

    validation_ids = set(
        splits.loc[splits["split"].eq("validation"), "row_id"]
    )
    eligible = songs[songs["row_id"].isin(validation_ids)]

    references = eligible.sample(
        n=min(10, len(eligible)), random_state=2026
    )
    rng = np.random.default_rng(2026)

    rating_rows = []
    ranking_rows = []

    for query_number, reference in enumerate(
        references.to_dict("records"), start=1
    ):
        query_id = f"Q{query_number:02d}"

        # Select a supported mood with at least five other candidates.
        moods = [
            mood
            for mood in sorted(reference["moods"])
            if songs["moods"].map(
                lambda values: mood in values
            ).sum() >= 6
        ]
        if not moods:
            raise ValueError(f"No eligible mood for {reference['title']}")

        mood = str(rng.choice(moods))

        options = {
            "title": reference["title"],
            "artist": reference["artist"],
            "mood": mood,
            "top_k": 5,
        }

        result_sets = {
            "baseline": recommend(songs, raw_features, **options),
            "autoencoder": recommend(songs, matrix, **options),
        }

        pooled = {}
        for results in result_sets.values():
            for candidate in results.to_dict("records"):
                pooled[int(candidate["row_id"])] = candidate

        # Random order hides each method's ranking.
        candidate_ids = sorted(pooled)
        rng.shuffle(candidate_ids)

        blind_ids = {}

        for number, candidate_id in enumerate(candidate_ids, start=1):
            candidate = pooled[candidate_id]
            item_id = f"{query_id}-C{number:02d}"
            blind_ids[candidate_id] = item_id

            rating_rows.append({
                "query_id": query_id,
                "item_id": item_id,
                "reference_title": reference["title"],
                "reference_artist": reference["artist"],
                "requested_mood": mood,
                "candidate_title": candidate["title"],
                "candidate_artist": candidate["artist"],
                "rating": "",
                "notes": "",
            })

        for method, results in result_sets.items():
            for rank, candidate in enumerate(
                results.to_dict("records"), start=1
            ):
                candidate_id = int(candidate["row_id"])
                ranking_rows.append({
                    "query_id": query_id,
                    "item_id": blind_ids[candidate_id],
                    "method": method,
                    "rank": rank,
                    "reference_row_id": reference["row_id"],
                    "candidate_row_id": candidate_id,
                })

    folder.mkdir(parents=True)
    pd.DataFrame(rating_rows).to_csv(
        folder / "ratings.csv", index=False
    )
    pd.DataFrame(ranking_rows).to_csv(
        folder / "method_key.csv", index=False
    )

    protocol = {
        "evaluation": "single-rater validation pilot",
        "seed": 2026,
        "queries": len(references),
        "recommendations_per_method": 5,
        "primary_metric": "mean Precision@5 across queries",
        "relevant_rating": 2,
        "rating_scale": {
            "0": "Poor fit",
            "1": "Somewhat suitable",
            "2": "Good fit for both reference song and requested mood",
        },
        "unknown": "Leave blank until listened to; never score unknown as zero",
        "comparison": "Paired comparison using the same queries and candidates",
    }
    (folder / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print(f"Evaluation queries: {len(references)}")
    print(f"Unique recommendation ratings needed: {len(rating_rows)}")
    print(f"Ratings file: {folder / 'ratings.csv'}")
    print("Do not open method_key.csv until ratings are complete.")


if __name__ == "__main__":
    main()