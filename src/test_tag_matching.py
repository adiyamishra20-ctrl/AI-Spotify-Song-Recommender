from pathlib import Path

import requests
from dotenv import dotenv_values

from src.enrich_catalog import fetch_tags


def main():
    root = Path(__file__).resolve().parents[1]
    api_key = dotenv_values(root / ".env").get("LASTFM_API_KEY")

    if not api_key:
        raise RuntimeError("API key missing.")

    cache_dir = root / "data" / "raw" / "lastfm_tags"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Two empty results and one previously successful control.
    songs = [
        ("Roxette", "It Must Have Been Love"),
        ("John Mayer", "Your Body Is a Wonderland"),
        ("Katy Perry", "Hot n Cold"),
    ]

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "SongRecommenderLearningProject/0.1"
        })

        for artist, title in songs:
            print(f"\n{title} — {artist}")

            variants = [
                ("original", artist, title),
                ("lowercase title", artist, title.lower()),
                ("lowercase both", artist.lower(), title.lower()),
            ]

            for label, query_artist, query_title in variants:
                try:
                    data, cached = fetch_tags(
                        session,
                        query_artist,
                        query_title,
                        api_key,
                        cache_dir,
                    )
                except LookupError:
                    print(f"  {label}: lookup error 6")
                    continue
                except RuntimeError as error:
                    print(f"Stopped: {error}")
                    return

                tags = data["toptags"]["tag"]
                names = ", ".join(tag["name"] for tag in tags[:5])
                source = "cache" if cached else "API"

                print(f"  {label}: {len(tags)} tags [{source}]")
                print(f"    {names or '(none)'}")
                print(f"    Returned identity: {data['toptags'].get('@attr', {})}")


if __name__ == "__main__":
    main()