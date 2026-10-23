import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def normalize_text(text):
    text = text.replace(r"\/", "/")
    text = text.replace("’", "'").replace("‘", "'")
    return comparison_key(text)


def base_title(title):
    # Remove a trailing guest credit, but retain remix/live descriptions.
    title = title.replace(r"\/", "/")
    title = re.sub(
        r"\s*[\(\[]\s*(?:w/|with\b|feat\.?|ft\.?|featuring\b)"
        r"\s+.*?[\)\]]\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return normalize_text(title)


def lead_artist(artist):
    # Do not split "and" or "&": they can be part of a band name.
    artist = re.split(
        r"\s+(?:featuring|feat\.?|ft\.?)\s+",
        artist,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return normalize_text(artist)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "external" / "dataset" / "yes_complete"
    reports = root / "reports"

    candidates = pd.read_csv(
        reports / "historical_matching_candidates.csv",
        keep_default_na=False,
    )

    tag_names = {}
    with (source / "tag_hash.txt").open(encoding="utf-8-sig") as file:
        for line in file:
            tag_id, name = line.rstrip("\r\n").split(",", 1)
            tag_names[int(tag_id)] = name.strip()

    tag_lines = (source / "tags.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()

    lookup = defaultdict(list)

    with (source / "song_hash.txt").open(encoding="utf-8-sig") as file:
        for line in file:
            song_id, title, artist = line.rstrip("\r\n").split("\t")
            song_id = int(song_id)

            value = tag_lines[song_id].strip()
            tags = (
                []
                if value in {"", "#"}
                else [tag_names[int(tag_id)] for tag_id in value.split()]
            )

            lookup[(base_title(title), lead_artist(artist))].append({
                "historical_song_id": song_id,
                "historical_title": title,
                "historical_artist": artist,
                "historical_tags_json": json.dumps(
                    tags, ensure_ascii=False
                ),
                "tag_count": len(tags),
            })

    proposals = []

    for song in candidates.to_dict("records"):
        key = (base_title(song["title"]), lead_artist(song["artist"]))
        matches = lookup.get(key, [])

        # Count all identity candidates, including those without tags.
        for match in matches:
            if match["tag_count"] == 0:
                continue

            proposals.append({
                "title": song["title"],
                "artist": song["artist"],
                **match,
                "candidate_count": len(matches),
                "review_decision": "pending",
            })

    columns = [
        "title", "artist", "historical_song_id",
        "historical_title", "historical_artist",
        "historical_tags_json", "tag_count",
        "candidate_count", "review_decision",
    ]
    result = pd.DataFrame(proposals, columns=columns)
    output = reports / "proposed_historical_matches.csv"

    # Preserve any review decisions from an earlier run.
    if output.exists():
        raise FileExistsError(
            f"{output.name} already exists. Rename it before regenerating."
        )

    result.to_csv(output, index=False)

    unique_songs = result[["title", "artist"]].drop_duplicates()
    print(f"Untagged candidates checked: {len(candidates)}")
    print(f"Songs with proposed tagged matches: {len(unique_songs)}")
    print(f"Total proposed matches: {len(result)}")

    print("\nFirst 15 proposals:")
    for row in result.head(15).to_dict("records"):
        print(f"\nChart: {row['title']} — {row['artist']}")
        print(
            f"Historical: {row['historical_title']} "
            f"— {row['historical_artist']}"
        )
        print(
            f"Tags: {row['tag_count']}; "
            f"identity candidates: {row['candidate_count']}"
        )

    print(f"\nSaved: {output}")
    print("Catalog unchanged; proposals require review.")


if __name__ == "__main__":
    main()