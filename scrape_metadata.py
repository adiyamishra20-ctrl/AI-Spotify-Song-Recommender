from bs4 import BeautifulSoup
import pandas as pd
import sys

if len(sys.argv) != 3:
    raise SystemExit("Usage: python scripts/scrape_metadata.py INPUT.html OUTPUT.csv")

with open(sys.argv[1], encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

rows = []
for card in soup.select("[data-track-name]"):
    rows.append({
        "track_name": card.get("data-track-name"),
        "artists": card.get("data-artists", ""),
        "genre": card.get("data-genre", ""),
        "mood": card.get("data-mood", "")
    })

pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(f"Extracted {len(rows)} records -> {sys.argv[2]}")
