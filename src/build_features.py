import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.audit_tags import GENRES, MOODS, normalize_tag


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"

    catalog = pd.read_csv(
        processed / "catalog_tags.csv",
        keep_default_na=False,
    )

    records = []

    for song in catalog.to_dict("records"):
        tags = {
            normalize_tag(tag)
            for tag in json.loads(song["tags_json"])
        }

        genres = sorted(tags & GENRES)
        moods = sorted(tags & MOODS)

        # Baseline recommendations require evidence for both.
        if not genres or not moods:
            continue

        records.append({
            "title": song["title"],
            "artist": song["artist"],
            "chart_year": song["chart_year"],
            "tag_source": song["tag_source"],
            "genres": genres,
            "moods": moods,
        })

    if not records:
        raise ValueError("No songs have both genre and mood annotations.")

    songs = pd.DataFrame(records)

    # Fixed vocabulary gives a consistent column order.
    genre_encoder = MultiLabelBinarizer(classes=sorted(GENRES))
    genre_features = genre_encoder.fit_transform(songs["genres"])

    mood_encoder = MultiLabelBinarizer(classes=sorted(MOODS))
    mood_features = mood_encoder.fit_transform(songs["moods"])

    genre_columns = [
        f"genre:{name}" for name in genre_encoder.classes_
    ]
    mood_columns = [
        f"mood:{name}" for name in mood_encoder.classes_
    ]

    features = pd.concat(
        [
            pd.DataFrame(genre_features, columns=genre_columns),
            pd.DataFrame(mood_features, columns=mood_columns),
        ],
        axis=1,
    )

    # Store explicit row IDs so metadata and feature rows can be aligned.
    songs.insert(0, "row_id", range(len(songs)))
    features.insert(0, "row_id", songs["row_id"])

    songs["genres"] = songs["genres"].map(
        lambda values: json.dumps(values, ensure_ascii=False)
    )
    songs["moods"] = songs["moods"].map(
        lambda values: json.dumps(values, ensure_ascii=False)
    )

    songs.to_csv(processed / "baseline_songs.csv", index=False)
    features.to_csv(processed / "baseline_features.csv", index=False)

    print(f"Baseline songs: {len(songs)}")
    print(f"Genre features: {len(genre_columns)}")
    print(f"Mood features: {len(mood_columns)}")
    print(f"Total features: {len(genre_columns) + len(mood_columns)}")

    print("\nFirst song:")
    print(songs.iloc[0][["title", "artist", "genres", "moods"]].to_string())

    print("\nSaved baseline_songs.csv and baseline_features.csv")


if __name__ == "__main__":
    main()