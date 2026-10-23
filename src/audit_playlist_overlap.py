import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def identity(title, artist):
    return comparison_key(title), comparison_key(artist)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "external" / "dataset" / "yes_complete"

    songs = pd.read_csv(
        root / "data" / "processed" / "baseline_songs.csv"
    )

    if songs["row_id"].duplicated().any():
        raise ValueError("Duplicate catalog row IDs.")

    # Start with conservative, unique exact-name matches.
    historical = defaultdict(list)

    with (source / "song_hash.txt").open(
        encoding="utf-8-sig"
    ) as file:
        for line in file:
            song_id, title, artist = line.rstrip("\r\n").split("\t")
            historical[identity(title, artist)].append(int(song_id))

    catalog = defaultdict(list)
    moods = {}

    for song in songs.to_dict("records"):
        row_id = int(song["row_id"])
        catalog[identity(song["title"], song["artist"])].append(row_id)
        moods[row_id] = set(json.loads(song["moods"]))

    mapping = {}

    for key, row_ids in catalog.items():
        source_ids = historical.get(key, [])

        if len(row_ids) == 1 and len(source_ids) == 1:
            mapping[source_ids[0]] = row_ids[0]

    transitions = 0
    mapped_transitions = 0
    shared_mood_transitions = 0
    unique_pairs = set()
    references = set()

    with (source / "train.txt").open(encoding="utf-8-sig") as file:
        external_ids = next(file).split()
        appearance_counts = next(file).split()

        if len(external_ids) != len(appearance_counts):
            raise ValueError("Playlist header lengths do not agree.")

        number_of_songs = len(external_ids)

        if mapping and max(mapping) >= number_of_songs:
            raise ValueError("Mapped IDs exceed the playlist song range.")

        for line_number, line in enumerate(file, start=3):
            playlist = [int(value) for value in line.split()]

            if any(
                song_id < 0 or song_id >= number_of_songs
                for song_id in playlist
            ):
                raise ValueError(f"Invalid song ID on line {line_number}")

            # Preserve actual adjacency. Do not remove unmatched songs first.
            for first, second in zip(playlist, playlist[1:]):
                transitions += 1

                if first not in mapping or second not in mapping:
                    continue

                reference = mapping[first]
                candidate = mapping[second]

                if reference == candidate:
                    continue

                mapped_transitions += 1
                unique_pairs.add((reference, candidate))
                references.add(reference)

                if moods[reference] & moods[candidate]:
                    shared_mood_transitions += 1

    report = {
        "catalog_songs": len(songs),
        "uniquely_mapped_songs": len(mapping),
        "training_transitions": transitions,
        "mapped_nonself_transitions": mapped_transitions,
        "unique_mapped_pairs": len(unique_pairs),
        "reference_songs_with_transitions": len(references),
        "mapped_transitions_sharing_a_dataset_mood": (
            shared_mood_transitions
        ),
    }

    reports = root / "reports"
    (reports / "playlist_overlap_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        [
            {"historical_song_id": source_id, "row_id": row_id}
            for source_id, row_id in sorted(mapping.items())
        ],
        columns=["historical_song_id", "row_id"],
    ).to_csv(reports / "playlist_song_mapping.csv", index=False)

    for name, value in report.items():
        print(f"{name}: {value}")

    print("\nTraining-data audit saved. Test playlists were not accessed.")


if __name__ == "__main__":
    main()