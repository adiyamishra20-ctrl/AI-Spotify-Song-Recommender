from bs4 import BeautifulSoup


def parse_songs(html: str) -> list[dict]:
    """Extract song metadata from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    songs = []

    for card in soup.select(".song"):
        title = card.select_one(".title")
        artist = card.select_one(".artist")

        # Skip incomplete entries instead of crashing.
        if title is None or artist is None:
            continue

        songs.append({
            "title": title.get_text(strip=True),
            "artist": artist.get_text(strip=True),
            "genre": card.get("data-genre"),
            "mood": card.get("data-mood"),
        })

    return songs


if __name__ == "__main__":
    sample_html = """
    <div class="song" data-genre="pop" data-mood="happy">
        <h2 class="title">Example Song A</h2>
        <span class="artist">Example Artist A</span>
    </div>
    <div class="song" data-genre="ambient" data-mood="calm">
        <h2 class="title">Example Song B</h2>
        <span class="artist">Example Artist B</span>
    </div>
    """

    for song in parse_songs(sample_html):
        print(song)