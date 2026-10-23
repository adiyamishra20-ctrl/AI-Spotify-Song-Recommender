from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]

    catalog = pd.read_csv(
        root / "data" / "processed" / "catalog_tags.csv",
        keep_default_na=False,
    )

    # Focus on older songs that the historical dataset could contain.
    missing = catalog[
        catalog["enrichment_status"].isin(
            ["no_tags", "lookup_error_6"]
        )
        & catalog["chart_year"].le(2011)
    ]

    print(f"Untagged candidates charting by 2011: {len(missing)}")

    collaborations = missing[
        missing["artist"].str.contains(
            r"\bfeaturing\b", case=False, regex=True
        )
    ]

    print(f"Of these, featuring credits: {len(collaborations)}")
    print("\nFirst 20 collaboration examples:")
    print(
        collaborations[["title", "artist"]]
        .head(20)
        .to_string(index=False)
    )

    output = root / "reports" / "historical_matching_candidates.csv"
    missing.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()