import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "external" / "dataset" / "yes_complete"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Read tag IDs and their names.
    tag_names = {}

    with (source / "tag_hash.txt").open(encoding="utf-8-sig") as file:
        for line in file:
            tag_id, name = line.rstrip("\r\n").split(",", 1)
            tag_names[int(tag_id)] = name.strip()

    # Preserve line positions: line number equals song ID.
    tag_lines = (source / "tags.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()

    song_tags = []

    for song_id, line in enumerate(tag_lines):
        value = line.strip()

        if value in {"", "#"}:
            song_tags.append([])
            continue

        ids = [int(value) for value in value.split()]
        unknown = set(ids) - tag_names.keys()

        if unknown:
            raise ValueError(
                f"Unknown tag IDs for song {song_id}: {unknown}"
            )

        song_tags.append([tag_names[tag_id] for tag_id in ids])

    # Keep all candidates for each identity so duplicates are detectable.
    lookup = defaultdict(list)
    seen_ids = set()

    with (source / "song_hash.txt").open(
        encoding="utf-8-sig"
    ) as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\r\n").split("\t")

            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 tab-separated fields on line {line_number}"
                )

            song_id_text, title, artist = parts
            song_id = int(song_id_text)

            if song_id in seen_ids or not 0 <= song_id < len(song_tags):
                raise ValueError(f"Invalid or duplicate song ID: {song_id}")

            seen_ids.add(song_id)

            key = (comparison_key(title), comparison_key(artist))
            lookup[key].append({
                "song_id": song_id,
                "tags": song_tags[song_id],
            })

    if seen_ids != set(range(len(song_tags))):
        raise ValueError("Song IDs and tag-file lines do not align.")

    catalog = pd.read_csv(
        root / "data" / "processed" / "catalog_cleaned_provisional.csv"
    )
    results = []

    for song in catalog.to_dict("records"):
        key = (
            comparison_key(song["title"]),
            comparison_key(song["artist"]),
        )
        candidates = lookup.get(key, [])

        if not candidates:
            status = "no_exact_match"
            tags = []
            source_id = ""

        elif len(candidates) > 1:
            status = "ambiguous_match"
            tags = []
            source_id = ""

        else:
            candidate = candidates[0]
            tags = candidate["tags"]
            source_id = candidate["song_id"]
            status = "matched_with_tags" if tags else "matched_no_tags"

        results.append({
            **song,
            "historical_status": status,
            "historical_song_id": source_id,
            "historical_tag_count": len(tags),
            "historical_tags_json": json.dumps(tags, ensure_ascii=False),
            "historical_source": (
                "Cornell Playlist Dataset / yes_complete"
                if len(candidates) == 1 else ""
            ),
        })

    result = pd.DataFrame(results)
    output = reports / "historical_tag_matches.csv"
    result.to_csv(output, index=False)

    print(f"Historical songs loaded: {len(seen_ids)}")
    print(f"Catalog candidates: {len(result)}")
    print("\nMatch results:")
    print(result["historical_status"].value_counts().to_string())

    tagged = result["historical_status"].eq("matched_with_tags")
    print(f"\nExact-match tag coverage: {tagged.mean():.1%}")

    print("\nFirst five tagged matches:")
    for song in result.loc[tagged].head(5).to_dict("records"):
        tags = json.loads(song["historical_tags_json"])
        print(f"{song['title']} — {song['artist']}")
        print("  Tags:", ", ".join(tags[:10]))

    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()