from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = (
        root / "data" / "external" / "dataset"
        / "yes_complete" / "train.txt"
    )

    playlist_count = 0
    appearances = 0
    transitions = 0
    self_transitions = 0
    empty_lines = 0
    examples = []

    with path.open(encoding="utf-8-sig") as file:
        external_ids = next(file).split()
        header_counts = [int(value) for value in next(file).split()]

        for line in file:
            playlist = [int(value) for value in line.split()]

            if not playlist:
                empty_lines += 1
                continue

            playlist_count += 1
            appearances += len(playlist)
            transitions += max(0, len(playlist) - 1)

            self_transitions += sum(
                first == second
                for first, second in zip(playlist, playlist[1:])
            )

            if len(examples) < 3:
                examples.append(playlist[:12])

    print("Header song IDs:", len(external_ids))
    print("Header appearance counts:", len(header_counts))
    print("Sum of header appearances:", sum(header_counts))
    print("Observed song appearances:", appearances)
    print("Nonempty playlists:", playlist_count)
    print("Empty lines:", empty_lines)
    print("All adjacent transitions:", transitions)
    print("Same-song transitions:", self_transitions)
    print("Different-song transitions:", transitions - self_transitions)

    print("\nFirst three playlist prefixes:")
    for example in examples:
        print(example)


if __name__ == "__main__":
    main()