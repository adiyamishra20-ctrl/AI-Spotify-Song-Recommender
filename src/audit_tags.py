import json
from collections import Counter
from pathlib import Path

import pandas as pd


# A deliberately small first-pass vocabulary.
GENRES = {
    "pop", "rock", "alternative", "alternative rock", "indie",
    "indie rock", "hip hop", "rap", "rnb", "soul", "country",
    "electronic", "electronica", "dance", "dance pop", "house",
    "techno", "trance", "jazz", "blues", "metal", "heavy metal",
    "punk", "punk rock", "folk", "reggae", "reggaeton",
    "disco", "funk", "grunge", "post grunge", "pop rock",
    "soft rock", "hard rock", "new wave", "synthpop",
    "electropop", "britpop", "gospel", "latin", "k pop",
}

MOODS = {
    "happy", "sad", "melancholic", "melancholy", "mellow",
    "relaxing", "calm", "chill", "chillout", "romantic",
    "angry", "aggressive", "energetic", "upbeat", "uplifting",
    "dark", "dreamy", "peaceful", "emotional", "nostalgic",
}

ALIASES = {
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "r&b": "rnb",
    "r & b": "rnb",
    "rhythm and blues": "rnb",
    "dance-pop": "dance pop",
    "post-grunge": "post grunge",
    "k-pop": "k pop",
}


def normalize_tag(tag):
    tag = " ".join(tag.casefold().split())
    return ALIASES.get(tag, tag)


def main():
    root = Path(__file__).resolve().parents[1]
    catalog = pd.read_csv(
        root / "data" / "processed" / "catalog_tags.csv",
        keep_default_na=False,
    )

    genre_counts = []
    mood_counts = []
    tag_counts = Counter()

    for row in catalog.to_dict("records"):
        tags = {
            normalize_tag(tag)
            for tag in json.loads(row["tags_json"])
        }

        tag_counts.update(tags)
        genre_counts.append(len(tags & GENRES))
        mood_counts.append(len(tags & MOODS))

    catalog["recognized_genre_count"] = genre_counts
    catalog["recognized_mood_count"] = mood_counts

    has_genre = catalog["recognized_genre_count"].gt(0)
    has_mood = catalog["recognized_mood_count"].gt(0)

    print(f"Total candidates: {len(catalog)}")
    print(f"With recognized genre tags: {has_genre.sum()}")
    print(f"With recognized mood tags: {has_mood.sum()}")
    print(f"With both: {(has_genre & has_mood).sum()}")

    print("\nCoverage by tag source:")
    summary = catalog.assign(
        has_genre=has_genre,
        has_mood=has_mood,
        has_both=has_genre & has_mood,
    )
    print(
        summary.groupby("tag_source")[
            ["has_genre", "has_mood", "has_both"]
        ].sum().to_string()
    )

    print("\nMost frequent tags:")
    for tag, count in tag_counts.most_common(40):
        print(f"{tag}: {count}")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    catalog.to_csv(reports / "tag_coverage_audit.csv", index=False)
    pd.DataFrame(
        tag_counts.most_common(),
        columns=["tag", "song_count"],
    ).to_csv(reports / "tag_frequencies.csv", index=False)

    print("\nSaved coverage audit and tag frequencies.")


if __name__ == "__main__":
    main()