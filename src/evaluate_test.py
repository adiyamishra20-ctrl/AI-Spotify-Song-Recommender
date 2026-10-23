import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline_recommender import load_data


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    playlist_path = (
        root / "data" / "external" / "dataset"
         / "yes_complete" / "test.txt"
    )

    songs, raw_features = load_data()
    embeddings = pd.read_csv(
        root / "data" / "processed" / "song_embeddings.csv"
    )
    mapping_table = pd.read_csv(reports / "playlist_song_mapping.csv")

    if not songs["row_id"].equals(embeddings["row_id"]):
        raise ValueError("Song and embedding rows do not align.")

    if (
        mapping_table["historical_song_id"].duplicated().any()
        or mapping_table["row_id"].duplicated().any()
    ):
        raise ValueError("Playlist mapping is not one-to-one.")

    embedding_matrix = embeddings[
        [f"embedding_{i}" for i in range(16)]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(embedding_matrix).all()
        or (np.linalg.norm(embedding_matrix, axis=1) < 1e-12).any()
    ):
        raise ValueError("Invalid embeddings.")
# Estimate the center using only the autoencoder's training songs.
    split = pd.read_csv(reports / "autoencoder_split.csv")

    if (
        split["row_id"].duplicated().any()
        or set(split["row_id"]) != set(songs["row_id"])
    ):
        raise ValueError("Training split does not align with songs.")

    training_ids = set(
        split.loc[split["split"].eq("train"), "row_id"]
    )
    training_mask = songs["row_id"].isin(training_ids).to_numpy()

    if not training_mask.any():
        raise ValueError("No training songs found.")

    center = embedding_matrix[training_mask].mean(axis=0)
    embedding_matrix = embedding_matrix - center

    if (np.linalg.norm(embedding_matrix, axis=1) < 1e-12).any():
        raise ValueError("Centering produced a zero-length embedding.")


    positions = {
        int(row_id): position
        for position, row_id in enumerate(songs["row_id"])
    }

    historical_to_position = {
        int(row.historical_song_id): positions[int(row.row_id)]
        for row in mapping_table.itertuples()
    }

    # Match candidates to the historical-era evaluation universe.
    universe = np.array(sorted(historical_to_position.values()))

    # One fixed mood per reference, chosen without looking at next songs.
    query_moods = {}
    candidate_positions = {}

    for position in universe:
        reference = songs.iloc[position]
        available = sorted(reference["moods"])

        identity = (
            f"{reference['title']}|{reference['artist']}|2026"
        ).encode("utf-8")
        number = int(hashlib.sha256(identity).hexdigest(), 16)
        mood = available[number % len(available)]

        eligible = np.array([
            candidate
            for candidate in universe
            if candidate != position
            and mood in songs.iloc[candidate]["moods"]
        ], dtype=int)

        if len(eligible) >= 10:
            query_moods[position] = mood
            candidate_positions[position] = eligible

    # Rank once per query using identical candidate pools.
    score_matrices = {
        "baseline": cosine_similarity(raw_features),
        "autoencoder": cosine_similarity(embedding_matrix),
    }

    ranks = {}
    for method, scores in score_matrices.items():
        ranks[method] = {}

        for position, eligible in candidate_positions.items():
            order = np.lexsort((
                songs.iloc[eligible]["row_id"].to_numpy(),
                -scores[position, eligible],
            ))
            ranks[method][position] = {
                int(candidate): rank
                for rank, candidate in enumerate(
                    eligible[order], start=1
                )
            }

    # Fixed 20% playlist subset for development.
    rng = np.random.default_rng(2026)
    rows = []
    selected_playlists = 0
    mapped_events = 0

    with playlist_path.open(encoding="utf-8-sig") as file:
        next(file)
        next(file)

        for line_number, line in enumerate(file, start=3):
            playlist = [int(value) for value in line.split()]
            if not playlist:
                continue


            selected_playlists += 1

            for first, second in zip(playlist, playlist[1:]):
                if (
                    first not in historical_to_position
                    or second not in historical_to_position
                ):
                    continue

                source = historical_to_position[first]
                target = historical_to_position[second]

                if source == target or source not in query_moods:
                    continue

                mapped_events += 1

                # Conditional task: next song must meet the fixed mood filter.
                if target not in ranks["baseline"][source]:
                    continue

                row = {
                    "playlist_line": line_number,
                    "reference_row_id": int(songs.iloc[source]["row_id"]),
                    "target_row_id": int(songs.iloc[target]["row_id"]),
                    "mood": query_moods[source],
                }

                for method in score_matrices:
                    rank = ranks[method][source][target]
                    row[f"{method}_rank"] = rank
                    # One observed relevant next song per event.
                    row[f"{method}_ndcg10"] = (
                        1 / np.log2(rank + 1) if rank <= 10 else 0.0
                    )

                rows.append(row)

    if not rows:
        raise ValueError("No eligible development events.")

    events = pd.DataFrame(rows)

    # Average within each reference first, so frequent references
    # do not dominate the overall score.
    metrics = ["baseline_ndcg10", "autoencoder_ndcg10"]
    per_reference = events.groupby("reference_row_id")[metrics].mean()
    scores = per_reference.mean()

    baseline = float(scores["baseline_ndcg10"])
    autoencoder = float(scores["autoencoder_ndcg10"])
    relative_gain = (
        100 * (autoencoder - baseline) / baseline
        if baseline > 0 else None
    )

    checksum = hashlib.sha256()
    with playlist_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)

    summary = {
        "stage": "held_out_test_centered_embeddings",
        "playlist_sha256": checksum.hexdigest(),
        "selected_playlists": selected_playlists,
        "mapped_events_before_target_mood_filter": mapped_events,
        "eligible_events": len(events),
        "evaluated_reference_songs": len(per_reference),
        "candidate_universe_size": len(universe),
        "baseline_ndcg10": baseline,
        "autoencoder_ndcg10": autoencoder,
        "relative_improvement_percent": relative_gain,
        "interpretation": (
            "Macro-average next-song NDCG@10, conditional on the "
            "observed target satisfying a preselected dataset mood."
        ),
    }

    events.to_csv(reports / "test_centered_events.csv", index=False)
    per_reference.to_csv(reports / "test_centered_per_reference.csv")
    (reports / "test_centered_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nHeld-out test complete. Do not tune on these results.")


if __name__ == "__main__":
    main()