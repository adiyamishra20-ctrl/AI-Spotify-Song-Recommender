from pathlib import Path
import unicodedata

import pandas as pd


def comparison_key(text):
    """Normalize case and whitespace for conservative duplicate detection."""
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.casefold().split())


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports"

    frames = []

    for year in range(1990, 2025):
        path = raw_dir / f"chart_{year}.csv"

        if not path.exists():
            raise ValueError(f"Missing chart file: {path.name}")

        frame = pd.read_csv(path)

        required = {"chart_year", "rank", "title", "artist", "source_url"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing required columns in {path.name}")

        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing values in {path.name}")

        if (
            len(frame) != 100
            or set(frame["rank"]) != set(range(1, 101))
            or not frame["chart_year"].eq(year).all()
        ):
            raise ValueError(f"Invalid chart data in {path.name}")

        for column in ["title", "artist", "source_url"]:
            if frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty {column} in {path.name}")

        frames.append(frame)

    entries = pd.concat(frames, ignore_index=True)
    entries = entries.sort_values(["chart_year", "rank"])

    # Keep original names; create separate keys for comparisons.
    entries["title_key"] = entries["title"].map(comparison_key)
    entries["artist_key"] = entries["artist"].map(comparison_key)

    keys = ["title_key", "artist_key"]
    repeated = entries.duplicated(subset=keys, keep=False)

    # Flag slashes for inspection, not automatic splitting.
    # A slash can also be part of a legitimate single-song title.
    needs_review = entries["title"].str.contains("/", regex=False)

    provisional = entries.drop_duplicates(subset=keys, keep="first")

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    entries.to_csv(processed_dir / "chart_entries.csv", index=False)
    provisional.to_csv(
        processed_dir / "catalog_provisional.csv", index=False
    )
    entries.loc[repeated].to_csv(
        reports_dir / "repeated_entries.csv", index=False
    )
    entries.loc[needs_review].to_csv(
        reports_dir / "titles_to_review.csv", index=False
    )

    print(f"Validated yearly files: {len(frames)}")
    print(f"Total chart entries: {len(entries)}")
    print(f"Provisional unique title/artist pairs: {len(provisional)}")
    print(f"Repeated appearances removed: {len(entries) - len(provisional)}")
    print(f"Entries with titles to review: {needs_review.sum()}")

    print("\nDistinct titles requiring review:")
    review = entries.loc[needs_review, ["title", "artist"]].drop_duplicates()
    print(review.to_string(index=False))


if __name__ == "__main__":
    main()