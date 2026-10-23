from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main():
    url = "https://www.last.fm/music/Coldplay/_/Yellow"

    response = requests.get(
        url,
        headers={"User-Agent": "SongRecommenderLearningProject/0.1"},
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_title = soup.title.get_text(strip=True) if soup.title else ""

    if "client challenge" in page_title.lower():
        print("Blocked: the website returned a JavaScript challenge.")
        return

    heading = soup.find("h1")

    if heading is None:
        print("Unexpected page: no song heading found.")
        return

    print("Song heading:", heading.get_text(strip=True))

    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "raw" / "yellow.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(response.text, encoding="utf-8")

    print("Saved page:", output)


if __name__ == "__main__":
    main()