import argparse
import csv
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main():
    parser = argparse.ArgumentParser(
        description="Scrape a yearly song chart."
    )
    parser.add_argument("--year", type=int, default=2000)
    args = parser.parse_args()
    year = args.year

    url = (
        "https://en.wikipedia.org/wiki/"
        f"Billboard_Year-End_Hot_100_singles_of_{year}"
    )

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cache_file = raw_dir / f"chart_{year}.html"

    if cache_file.exists():
        html = cache_file.read_text(encoding="utf-8")
        print(f"Using cached page for {year}")
    else:
        # Space out requests when running several years.
        time.sleep(2)

        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "SongRecommenderLearningProject/0.1"
                },
                timeout=30,
            )
        except requests.RequestException:
            print(f"Download failed for {year}. Check your connection.")
            return

        if response.status_code != 200:
            print(f"Download failed: HTTP {response.status_code}")
            return

        html = response.text

    soup = BeautifulSoup(html, "html.parser")

    # Find the chart using its column names.
    chart = None

    for table in soup.select("table.wikitable"):
        first_row = table.find("tr")

        if first_row is None:
            continue

        headers = [
            cell.get_text(" ", strip=True).lower()
            for cell in first_row.find_all(["th", "td"])
        ]

        if (
            len(headers) == 3
            and headers[0] in {"no.", "no", "№", "rank", "#"}
            and headers[1] == "title"
            and headers[2] in {"artist(s)", "artist", "artists"}
        ):
            chart = table
            break

    if chart is None:
        print("Chart table missing. The page may have changed or been blocked.")
        page_title = (
            soup.title.get_text(strip=True) if soup.title else "Missing"
        )
        print("Page title:", page_title)
        return

    # Remove citation numbers before extracting text.
    for reference in chart.select("sup.reference"):
        reference.decompose()

    songs = []
    shared_artist = None
    artist_rows_remaining = 0

    for row in chart.find_all("tr")[1:]:
        cells = row.find_all(["th", "td"], recursive=False)

        if len(cells) == 3:
            artist_cell = cells[2]
            artist = artist_cell.get_text(" ", strip=True)

            # Remember artists whose cell spans multiple rows.
            shared_artist = artist
            artist_rows_remaining = int(
                artist_cell.get("rowspan", "1")
            ) - 1

        elif len(cells) == 2 and artist_rows_remaining > 0:
            artist = shared_artist
            artist_rows_remaining -= 1

        else:
            raise ValueError(f"Unexpected table row: {row}")

        rank = cells[0].get_text(strip=True)
        title = cells[1].get_text(" ", strip=True).strip('"“” ')

        if not rank.isdigit() or not title or not artist:
            raise ValueError(f"Invalid song fields: {row}")

        songs.append({
            "chart_year": year,
            "rank": int(rank),
            "title": title,
            "artist": artist,
            "source_url": url,
        })

    found_ranks = {song["rank"] for song in songs}
    expected_ranks = set(range(1, 101))

    print("Missing ranks:", sorted(expected_ranks - found_ranks))

    # Check completeness before saving.
    if len(songs) != 100 or found_ranks != expected_ranks:
        print(f"Validation failed: extracted {len(songs)} rows.")
        print("Expected 100 songs with unique ranks from 1 to 100.")
        return

    # Cache only pages that passed validation.
    cache_file.write_text(html, encoding="utf-8")

    output = raw_dir / f"chart_{year}.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(songs[0]))
        writer.writeheader()
        writer.writerows(songs)

    print(f"Saved {len(songs)} songs to {output}")

    for song in songs[:5]:
        print(f"{song['rank']}. {song['title']} — {song['artist']}")


if __name__ == "__main__":
    main()



