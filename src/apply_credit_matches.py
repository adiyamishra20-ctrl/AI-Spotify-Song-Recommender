import json
import re
from pathlib import Path

import pandas as pd

from src.prepare_catalog import comparison_key


def normalize(text):
    text = text.replace(r"\/", "/")
    text = text.replace("’", "'").replace("‘", "'")
    return comparison_key(text)


def credit_identity(title, artist):
    """Return title, lead artist, and explicit guest credit."""
    title = title.replace(r"\/", "/")
    artist_parts = re.split(
        r"\s+(?:featuring|feat\.?|ft\.?)\s+",
        artist,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    lead = artist_parts[0]
    artist_guest = artist_parts[1] if len(artist_parts) == 2 else ""

    suffix = re.search(
        r"\s*[\(\[]\s*(?:w/|with\b|feat\.?|ft\.?|featuring\b)"
        r"\s+(.+?)[\)\]]\s*$",
        title,
        flags=re.IGNORECASE,
    )

    title_guest = ""
    if suffix:
        title_guest = suffix.group(1)
        title = title[:suffix.start()]

    # Reject conflicting credits rather than choosing one.
    if (
        artist_guest and title_guest
        and normalize(artist_guest) != normalize(title_guest)
    ):
        return None

    guest = artist_guest or title_guest

    return normalize(title), normalize(lead), normalize(guest)


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    path = root / "data" / "processed" / "catalog_tags.csv"

    catalog = pd.read_csv(path, keep_default_na=False)
    proposals = pd.read_csv(
        reports / "proposed_historical_matches.csv",
        keep_default_na=False,
    )

    if "credit_match_source_id" not in catalog:
        catalog["credit_match_source_id"] = ""

    decisions = []
    recovered = 0

    for (title, artist), group in proposals.groupby(
        ["title", "artist"], sort=False
    ):
        decision = "needs_review"
        reason = "Multiple historical candidates"

        if len(group) == 1 and int(group.iloc[0]["candidate_count"]) == 1:
            proposal = group.iloc[0]

            chart_credit = credit_identity(title, artist)
            historical_credit = credit_identity(
                proposal["historical_title"],
                proposal["historical_artist"],
            )

            if chart_credit is None or historical_credit is None:
                reason = "Conflicting explicit guest credits"

            elif chart_credit != historical_credit:
                reason = "Title, lead artist, or guest credits differ"

            else:
                matches = catalog.index[
                    catalog["title"].eq(title)
                    & catalog["artist"].eq(artist)
                ]

                if len(matches) != 1:
                    raise ValueError(f"Catalog identity not unique: {title}")

                index = matches[0]
                tags = json.loads(proposal["historical_tags_json"])

                if not tags:
                    reason = "Historical candidate has no tags"

                elif json.loads(catalog.at[index, "tags_json"]):
                    decision = "already_tagged"
                    reason = "Existing tags preserved"

                else:
                    catalog.at[index, "tags_json"] = json.dumps(
                        tags, ensure_ascii=False
                    )
                    catalog.at[index, "tag_source"] = "cornell_lastfm"
                    catalog.at[index, "enrichment_status"] = "historical_tags"
                    catalog.at[index, "credit_match_source_id"] = str(
                        proposal["historical_song_id"]
                    )

                    decision = "accepted"
                    reason = "Unique candidate; normalized full credits agree"
                    recovered += 1

        decisions.append({
            "title": title,
            "artist": artist,
            "decision": decision,
            "reason": reason,
        })

    if recovered:
        backup = path.with_name("catalog_tags_before_credit_matches.csv")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())

        temporary = path.with_suffix(".tmp.csv")
        catalog.to_csv(temporary, index=False)
        temporary.replace(path)

    report = pd.DataFrame(decisions)
    report.to_csv(reports / "credit_match_decisions.csv", index=False)

    tagged = catalog["tags_json"].map(
        lambda value: bool(json.loads(value))
    )

    print(f"Newly recovered songs: {recovered}")
    print(f"Candidates with tags: {tagged.sum()}/{len(catalog)}")
    print("\nReview outcomes:")
    print(report["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()