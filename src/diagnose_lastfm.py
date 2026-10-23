from pathlib import Path
import json

import requests
from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    api_key = dotenv_values(root / ".env").get("LASTFM_API_KEY")

    if not api_key:
        print("API key missing.")
        return

    with requests.Session() as session:
        for method in ["track.getTopTags", "track.getInfo"]:
            print(f"\n--- {method} ---")

            try:
                response = session.get(
                    "https://ws.audioscrobbler.com/2.0/",
                    params={
                        "method": method,
                        "artist": "Roxette",
                        "track": "It Must Have Been Love",
                        "api_key": api_key,
                        "format": "json",
                        "autocorrect": 1,
                    },
                    timeout=30,
                )

                print("HTTP status:", response.status_code)

                if response.status_code != 200:
                    return

                data = response.json()

            except requests.RequestException:
                print("Network request failed.")
                return
            except ValueError:
                print("Response was not valid JSON.")
                return

            if "error" in data:
                print("API error code:", data["error"])
                return

            if method == "track.getTopTags":
                print(json.dumps(
                    data.get("toptags", {}),
                    ensure_ascii=False,
                    indent=2,
                ))
            else:
                track = data.get("track", {})
                details = {
                    "name": track.get("name"),
                    "artist": track.get("artist"),
                    "url": track.get("url"),
                    "listeners": track.get("listeners"),
                    "toptags": track.get("toptags"),
                }
                print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()