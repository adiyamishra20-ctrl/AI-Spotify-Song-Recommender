from pathlib import Path

import requests
from dotenv import dotenv_values


def main():
    project_root = Path(__file__).resolve().parents[1]
    api_key = dotenv_values(project_root / ".env").get("LASTFM_API_KEY")

    if not api_key:
        print("API key missing from the project's .env file.")
        return

    params = {
        "method": "track.getTopTags",
        "artist": "Coldplay",
        "track": "Yellow",
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
    }

    try:
        response = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            print(f"HTTP error: {response.status_code}")
            return

        data = response.json()

    except requests.RequestException:
        print("Request failed. Check your internet connection and try later.")
        return
    except ValueError:
        print("The server did not return valid JSON.")
        return

    if "error" in data:
        print("Last.fm error:", data.get("message", "Unknown error"))
        return

    tags = data.get("toptags", {}).get("tag", [])

    if not tags:
        print("No tags found for this song.")
        return

    print("Tags for Yellow — Coldplay:")
    for tag in tags[:10]:
        print(f"- {tag['name']} (weight: {tag['count']})")


if __name__ == "__main__":
    main()