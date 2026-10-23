import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values


def fetch_tags(session, artist, title, api_key, cache_dir):
    # A stable filename based on the exact artist/title combination.
    identity = json.dumps([artist, title], ensure_ascii=False)
    cache_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{cache_id}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8")), True

    time.sleep(1)

    try:
        response = session.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getTopTags",
                "artist": artist,
                "track": title,
                "api_key": api_key,
                "format": "json",
                "autocorrect": 1,
            },
            timeout=30,
        )
    except requests.RequestException:
        # Avoid printing exception details that could contain the API key.
        raise RuntimeError("Network request failed.") from None

    if response.status_code != 200:
        raise RuntimeError(f"HTTP error: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Server returned invalid JSON.") from None

    if "error" in data:
        if str(data["error"]) == "6":
            raise LookupError("Last.fm rejected this song lookup (code 6).")

        raise RuntimeError(f"Last.fm API error code: {data['error']}")

    top_tags = data.get("toptags")
    if not isinstance(top_tags, dict):
        raise RuntimeError("Unexpected response: missing toptags object.")

    if not isinstance(top_tags.get("tag"), list):
        raise RuntimeError("Unexpected response: missing tag list.")

    # Store successful responses only.
    cache_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data, False


def main():
    root = Path(__file__).resolve().parents[1]
    api_key = dotenv_values(root / ".env").get("LASTFM_API_KEY")

    if not api_key:
        raise RuntimeError("API key missing from .env.")

    catalog = pd.read_csv(
        root / "data" / "processed" / "catalog_cleaned_provisional.csv"
    )

    cache_dir = root / "data" / "raw" / "lastfm_tags"
    cache_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "SongRecommenderLearningProject/0.1"
        })

                # Sample across the catalog instead of only the first chart year.
        sample = catalog.sample(n=min(50, len(catalog)), random_state=42)

        for song in sample.to_dict("records"):
            artist = song["artist"]
            title = song["title"]

            try:
                data, cached = fetch_tags(
                    session, artist, title, api_key, cache_dir
                )
            except LookupError as error:
                results.append({
                    "title": title,
                    "artist": artist,
                    "returned_track": "",
                    "returned_artist": "",
                    "tag_count": 0,
                    "tags_json": "[]",
                    "status": "lookup_error_6",
                })
                print(f"\nLookup failed: {title} — {artist}: {error}")
                continue

            tags = data["toptags"]["tag"]
            tag_names = [tag["name"] for tag in tags]

            # Record any identity information returned by the API.
            returned = data["toptags"].get("@attr", {})

            results.append({
                "title": title,
                "artist": artist,
                "returned_track": returned.get("track", ""),
                "returned_artist": returned.get("artist", ""),
                "tag_count": len(tags),
                "tags_json": json.dumps(tags, ensure_ascii=False),
                "status": "tags_found" if tags else "no_tags",
            })

            source = "cache" if cached else "API"
            print(f"\n{title} — {artist} [{source}]")
            print("Top tags:", ", ".join(tag_names[:8]) or "(none)")

    if results:
        output = reports_dir / "enrichment_sample_50.csv"
        pd.DataFrame(results).to_csv(output, index=False)

        tagged = sum(row["tag_count"] > 0 for row in results)
        print(f"\nSongs with tags: {tagged}/{len(results)}")
        print(f"Saved pilot report: {output}")


if __name__ == "__main__":
    main()