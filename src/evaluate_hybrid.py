from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline_recommender import load_data


WEIGHTS = [0.0, 0.10, 0.25, 0.50, 1.0]


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    output = reports / "hybrid_development_results.csv"

    if output.exists():
        raise FileExistsError(
            "Hybrid results already exist. Preserve this bounded experiment."
        )

    songs, raw = load_data()
    embeddings = pd.read_csv(
        root / "data" / "processed" / "song_embeddings.csv"
    )
    split = pd.read_csv(reports / "autoencoder_split.csv")
    mapping = pd.read_csv(reports / "playlist_song_mapping.csv")
    events = pd.read_csv(reports / "development_ranking_events.csv")

    if events.empty:
        raise ValueError("No saved development events.")

    if (
        embeddings["row_id"].duplicated().any()
        or not songs["row_id"].equals(embeddings["row_id"])
    ):
        raise ValueError("Embedding rows do not align.")

    if (
        split["row_id"].duplicated().any()
        or set(split["row_id"]) != set(songs["row_id"])
    ):
        raise ValueError("Training split does not align.")

    if (
        mapping["historical_song_id"].duplicated().any()
        or mapping["row_id"].duplicated().any()
    ):
        raise ValueError("Playlist mapping is not one-to-one.")

    matrix = embeddings[
        [f"embedding_{i}" for i in range(16)]
    ].to_numpy(dtype=float)

    training_ids = set(split.loc[split["split"].eq("train"), "row_id"])
    training_mask = songs["row_id"].isin(training_ids).to_numpy()

    if not training_mask.any():
        raise ValueError("No training rows.")

    matrix -= matrix[training_mask].mean(axis=0)

    if (
        not np.isfinite(matrix).all()
        or (np.linalg.norm(matrix, axis=1) < 1e-12).any()
    ):
        raise ValueError("Invalid centered embeddings.")

    positions = {
        int(row_id): position
        for position, row_id in enumerate(songs["row_id"])
    }

    universe = np.array(sorted(
        positions[int(row_id)] for row_id in mapping["row_id"]
    ))

    raw_scores = cosine_similarity(raw)
    embedding_scores = cosine_similarity(matrix)

    # Each saved development event has the same query mood and target
    # used in the original development evaluation.
    queries = {}

    for (reference_id, mood), group in events.groupby(
        ["reference_row_id", "mood"], sort=True
    ):
        reference_position = positions[int(reference_id)]

        eligible = np.array([
            candidate for candidate in universe
            if candidate != reference_position
            and mood in songs.iloc[candidate]["moods"]
        ], dtype=int)

        if len(eligible) < 10:
            raise ValueError("Candidate pool changed.")

        target_positions = np.array([
            positions[int(row_id)] for row_id in group["target_row_id"]
        ])

        if not np.isin(target_positions, eligible).all():
            raise ValueError("An event target is outside its candidate pool.")

        queries[(int(reference_id), mood)] = (
            group.index.to_numpy(),
            reference_position,
            eligible,
            target_positions,
        )

    results = []
    details = []

    for weight in WEIGHTS:
        scores = (
            (1.0 - weight) * raw_scores
            + weight * embedding_scores
        )
        event_values = np.zeros(len(events), dtype=float)

        for indices, reference, eligible, targets in queries.values():
            order = np.lexsort((
                songs.iloc[eligible]["row_id"].to_numpy(),
                -scores[reference, eligible],
            ))

            ranks = np.zeros(len(songs), dtype=int)
            ranks[eligible[order]] = np.arange(1, len(eligible) + 1)

            target_ranks = ranks[targets]
            values = np.zeros(len(targets), dtype=float)
            hit = target_ranks <= 10
            values[hit] = 1.0 / np.log2(target_ranks[hit] + 1)
            event_values[indices] = values

        measured = pd.DataFrame({
            "reference_row_id": events["reference_row_id"],
            "ndcg10": event_values,
        })
        per_reference = measured.groupby("reference_row_id")["ndcg10"].mean()
        score = float(per_reference.mean())

        results.append({
            "embedding_weight": weight,
            "raw_feature_weight": 1.0 - weight,
            "development_ndcg10": score,
        })

        for reference_id, value in per_reference.items():
            details.append({
                "reference_row_id": reference_id,
                "embedding_weight": weight,
                "ndcg10": value,
            })

        print(
            f"Embedding weight {weight:.0%}: NDCG@10 = {score:.8f}",
            flush=True,
        )

    result = pd.DataFrame(results)
    baseline = float(result.iloc[0]["development_ndcg10"])

    # Ensure the endpoints reproduce the earlier experiments.
    original = json.loads(
        (reports / "development_metrics.json").read_text(encoding="utf-8")
    )
    centered = json.loads(
        (reports / "development_centered_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    if not np.isclose(
        baseline, original["baseline_ndcg10"], rtol=1e-8, atol=1e-10
    ):
        raise ValueError("Baseline changed. Results were not saved.")

    if not np.isclose(
        result.iloc[-1]["development_ndcg10"],
        centered["autoencoder_ndcg10"],
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Centered-model endpoint changed. Results not saved.")

    result["relative_change_percent"] = (
        100 * (result["development_ndcg10"] - baseline) / baseline
        if baseline > 0 else np.nan
    )

    # On an exact tie, prefer the lower embedding weight.
    best = result.sort_values(
        ["development_ndcg10", "embedding_weight"],
        ascending=[False, True],
    ).iloc[0]

    result.to_csv(output, index=False)
    pd.DataFrame(details).to_csv(
        reports / "hybrid_development_per_reference.csv", index=False
    )

    summary = {
        "stage": "development_only",
        "weights_tested": WEIGHTS,
        "eligible_events": len(events),
        "reference_songs": events["reference_row_id"].nunique(),
        "best_embedding_weight": float(best["embedding_weight"]),
        "best_development_ndcg10": float(best["development_ndcg10"]),
        "relative_change_percent": float(best["relative_change_percent"]),
        "test_playlists_accessed": False,
    }
    (reports / "hybrid_development_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nResults:")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n" + json.dumps(summary, indent=2))
    print("\nBounded experiment complete. No app settings were changed.")


if __name__ == "__main__":
    main()