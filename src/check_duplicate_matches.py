import json
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def tag_signature(value):
    tags = json.loads(value)
    return tuple(sorted({
        comparison_key(tag)
        for tag in tags
    }))


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "reports" / "proposed_historical_matches.csv"
    proposals = pd.read_csv(source, keep_default_na=False)

    results = []

    for (title, artist), group in proposals.groupby(
        ["title", "artist"], sort=False
    ):
        if int(group["candidate_count"].max()) <= 1:
            continue

        chart_identity = (
            comparison_key(title),
            comparison_key(artist),
        )

        identities = {
            (
                comparison_key(row["historical_title"]),
                comparison_key(row["historical_artist"]),
            )
            for row in group.to_dict("records")
        }

        signatures = {
            tag_signature(value)
            for value in group["historical_tags_json"]
        }

        # Proposals omit untagged candidates, so check completeness too.
        all_candidates_present = (
            group["historical_song_id"].nunique()
            == int(group["candidate_count"].max())
        )

        agrees = (
            all_candidates_present
            and identities == {chart_identity}
            and len(signatures) == 1
        )

        results.append({
            "title": title,
            "artist": artist,
            "candidate_count": int(group["candidate_count"].max()),
            "all_candidates_present": all_candidates_present,
            "exact_identity_and_tags_agree": agrees,
        })

    columns = [
        "title", "artist", "candidate_count",
        "all_candidates_present", "exact_identity_and_tags_agree",
    ]
    report = pd.DataFrame(results, columns=columns)
    output = root / "reports" / "duplicate_match_audit.csv"
    report.to_csv(output, index=False)

    print(f"Songs with multiple candidates: {len(report)}")
    print(
        "Exact identities with identical tag sets:",
        int(report["exact_identity_and_tags_agree"].sum()),
    )
    print("\nDetails:")
    print(report.to_string(index=False))
    print(f"\nSaved: {output}")
    print("Catalog unchanged.")


if __name__ == "__main__":
    main()