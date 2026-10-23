import re
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    reports = root / "reports"

    entries = pd.read_csv(processed / "chart_entries.csv")
    cleaned = []
    unresolved = []

    for entry in entries.to_dict("records"):
        raw_title = entry["title"]
        raw_artist = entry["artist"]

        # Split only when a closing quote precedes the slash.
        # Examples: Song A " / " Song B, Song A" / "Song B
        titles = [
            part.strip(' "“”')
            for part in re.split(r'["”]\s*/\s*', raw_title)
        ]

        if len(titles) == 1:
            # Preserve slash titles until we verify their meaning.
            artists = [raw_artist]
            status = "review_slash" if "/" in raw_title else "unchanged"

        elif (
            titles == ["Scream", "Childhood"]
            and raw_artist == "Michael Jackson and Janet Jackson"
        ):
            artists = [
                "Michael Jackson and Janet Jackson",
                "Michael Jackson",
            ]
            status = "split_with_verified_artist_credits"

        elif " / " in raw_artist:
            artists = [
                part.strip() for part in raw_artist.split(" / ")
            ]
            status = "split_titles_and_artists"

        else:
            artists = [raw_artist] * len(titles)
            status = "split_titles_shared_artist"

        if len(titles) != len(artists) or not all(titles + artists):
            unresolved.append({
                **entry,
                "reason": "Title and artist components do not align",
            })
            continue

        for title, artist in zip(titles, artists):
            cleaned.append({
                "chart_year": entry["chart_year"],
                "rank": entry["rank"],
                "title": title,
                "artist": artist,
                "raw_title": raw_title,
                "raw_artist": raw_artist,
                "source_url": entry["source_url"],
                "cleaning_status": status,
                "title_key": comparison_key(title),
                "artist_key": comparison_key(artist),
            })

    appearances = pd.DataFrame(cleaned)
    appearances = appearances.sort_values(["chart_year", "rank"])

    catalog = appearances.drop_duplicates(
        subset=["title_key", "artist_key"],
        keep="first",
    )

    # Preserve every chart appearance, including split components.
    appearances.to_csv(
        processed / "song_appearances.csv", index=False
    )
    catalog.to_csv(
        processed / "catalog_cleaned_provisional.csv", index=False
    )

    unresolved_columns = list(entries.columns) + ["reason"]
    pd.DataFrame(unresolved, columns=unresolved_columns).to_csv(
        reports / "unresolved_chart_entries.csv", index=False
    )

    slash_review = catalog[
        catalog["cleaning_status"].eq("review_slash")
    ]
    slash_review.to_csv(
        reports / "remaining_slash_titles.csv", index=False
    )

    print(f"Original chart entries: {len(entries)}")
    print(f"Song appearances after splitting: {len(appearances)}")
    print(f"Provisional unique songs: {len(catalog)}")
    print(f"Unresolved chart entries: {len(unresolved)}")
    print(f"Remaining slash titles: {len(slash_review)}")

    print("\nSlash titles preserved for review:")
    print(slash_review[["title", "artist"]].to_string(index=False))


if __name__ == "__main__":
    main()

