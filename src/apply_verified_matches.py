import json
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def identity(title, artist):
    return comparison_key(title), comparison_key(artist)


def tag_set(value):
    return {
        comparison_key(tag)
        for tag in json.loads(value)
    }


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    catalog_path = root / "data" / "processed" / "catalog_tags.csv"

    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    proposals = pd.read_csv(
        reports / "proposed_historical_matches.csv",
        keep_default_na=False,
    )
    audit = pd.read_csv(
        reports / "duplicate_match_audit.csv",
        keep_default_na=False,
    )

    accepted = audit[
        audit["exact_identity_and_tags_agree"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]

    catalog_keys = [
        identity(title, artist)
        for title, artist in zip(catalog["title"], catalog["artist"])
    ]

    if "historical_consensus_ids" not in catalog.columns:
        catalog["historical_consensus_ids"] = ""

    changes = []

    for decision in accepted.to_dict("records"):
        key = identity(decision["title"], decision["artist"])

        group = proposals[
            proposals["title"].eq(decision["title"])
            & proposals["artist"].eq(decision["artist"])
        ]

        # Recheck the evidence before applying the saved audit decision.
        source_ids = sorted(
            group["historical_song_id"].astype(int).unique().tolist()
        )
        expected = int(decision["candidate_count"])

        if len(source_ids) != expected:
            raise ValueError(f"Incomplete evidence for {key}")

        signatures = {
            frozenset(tag_set(value))
            for value in group["historical_tags_json"]
        }
        source_identities = {
            identity(row["historical_title"], row["historical_artist"])
            for row in group.to_dict("records")
        }

        if (
            source_identities != {key}
            or len(signatures) != 1
            or not next(iter(signatures))
        ):
            raise ValueError(f"Evidence no longer agrees for {key}")

        matches = [
            index for index, candidate in enumerate(catalog_keys)
            if candidate == key
        ]
        if len(matches) != 1:
            raise ValueError(f"Catalog identity is not unique: {key}")

        index = matches[0]

        # Preserve any tags already attached; rerunning is safe.
        if json.loads(catalog.at[index, "tags_json"]):
            continue

        catalog.at[index, "tags_json"] = (
            group.iloc[0]["historical_tags_json"]
        )
        catalog.at[index, "tag_source"] = "cornell_lastfm"
        catalog.at[index, "enrichment_status"] = "historical_tags"
        catalog.at[index, "historical_consensus_ids"] = json.dumps(
            source_ids
        )

        changes.append({
            "title": decision["title"],
            "artist": decision["artist"],
            "source_ids": source_ids,
        })

    if changes:
        # Preserve the pre-update catalog once.
        backup = catalog_path.with_name("catalog_tags_before_consensus.csv")
        if not backup.exists():
            backup.write_bytes(catalog_path.read_bytes())

        temporary = catalog_path.with_suffix(".tmp.csv")
        catalog.to_csv(temporary, index=False)
        temporary.replace(catalog_path)

    tagged = catalog["tags_json"].map(
        lambda value: bool(json.loads(value))
    )

    print(f"Newly recovered songs: {len(changes)}")
    print(f"Candidates with tags: {tagged.sum()}/{len(catalog)}")

    for change in changes:
        print(f"- {change['title']} — {change['artist']}")

    print("Other ambiguous and guest-credit matches remain unchanged.")


if __name__ == "__main__":
    main()