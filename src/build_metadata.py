import argparse
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

from src.enrich_catalog import fetch_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum pending songs to process; 0 means all.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be zero or positive.")

    root = Path(__file__).resolve().parents[1]
    source = root / "reports" / "historical_tag_matches.csv"
    output = root / "data" / "processed" / "catalog_tags.csv"
    cache_dir = root / "data" / "raw" / "lastfm_tags"

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_key = dotenv_values(root / ".env").get("LASTFM_API_KEY")
    if not api_key:
        raise RuntimeError("API key missing from .env.")

    if output.exists():
        catalog = pd.read_csv(output, keep_default_na=False)
        print("Resuming saved progress.", flush=True)

    else:
        catalog = pd.read_csv(source, keep_default_na=False)

        catalog["tags_json"] = "[]"
        catalog["tag_source"] = ""
        catalog["enrichment_status"] = "pending"
        catalog["api_returned_artist"] = ""
        catalog["api_returned_track"] = ""

        historical = catalog["historical_status"].eq(
            "matched_with_tags"
        )

        catalog.loc[historical, "tags_json"] = catalog.loc[
            historical, "historical_tags_json"
        ]
        catalog.loc[historical, "tag_source"] = "cornell_lastfm"
        catalog.loc[historical, "enrichment_status"] = "historical_tags"

        print(
            f"Loaded {len(catalog)} candidates; "
            f"{historical.sum()} already have historical tags.",
            flush=True,
        )

    def save_checkpoint():
        temporary = output.with_suffix(".tmp.csv")
        catalog.to_csv(temporary, index=False)
        temporary.replace(output)

    pending = catalog.index[
        catalog["enrichment_status"].eq("pending")
    ].tolist()

    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Processing up to {len(pending)} songs this run.", flush=True)

    # Verify we can save before making API requests.
    save_checkpoint()

    try:
        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "SongRecommenderLearningProject/0.1"
            })

            for count, index in enumerate(pending, start=1):
                title = catalog.at[index, "title"]
                artist = catalog.at[index, "artist"]

                try:
                    data, cached = fetch_tags(
                        session,
                        artist,
                        title,
                        api_key,
                        cache_dir,
                    )

                except LookupError:
                    catalog.at[index, "enrichment_status"] = (
                        "lookup_error_6"
                    )
                    print(
                        f"[{count}/{len(pending)}] {title}: lookup failed",
                        flush=True,
                    )

                except RuntimeError as error:
                    print(f"Stopped at {title}: {error}", flush=True)
                    # Keep this song pending for a later retry.
                    break

                else:
                    tags = data["toptags"]["tag"]
                    names = list(dict.fromkeys(
                        tag["name"].strip()
                        for tag in tags
                        if tag.get("name", "").strip()
                    ))

                    identity = data["toptags"].get("@attr", {})

                    catalog.at[index, "api_returned_artist"] = (
                        identity.get("artist", "")
                    )
                    catalog.at[index, "api_returned_track"] = (
                        identity.get("track", "")
                    )
                    catalog.at[index, "tags_json"] = json.dumps(
                        names, ensure_ascii=False
                    )
                    catalog.at[index, "tag_source"] = "lastfm_api"
                    catalog.at[index, "enrichment_status"] = (
                        "api_tags" if names else "no_tags"
                    )

                    origin = "cache" if cached else "API"
                    print(
                        f"[{count}/{len(pending)}] {title}: "
                        f"{len(names)} tags [{origin}]",
                        flush=True,
                    )

                if count % 25 == 0:
                    save_checkpoint()
                    print("Progress saved.", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted; saving completed results.", flush=True)

    finally:
        save_checkpoint()

    tagged = catalog["enrichment_status"].isin(
        ["historical_tags", "api_tags"]
    )

    print("\nCatalog status:")
    print(catalog["enrichment_status"].value_counts().to_string())
    print(f"\nCandidates with tags: {tagged.sum()}/{len(catalog)}")
    print(f"Tag coverage: {tagged.mean():.1%}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()