from pathlib import Path

import requests
from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    key = dotenv_values(root / ".env").get("GROQ_API_KEY")

    if not key:
        print("GROQ_API_KEY is missing.")
        return

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
    except requests.RequestException:
        print("Network request failed.")
        return

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Could not retrieve available models.")
        return

    try:
        models = response.json()["data"]
        model_ids = sorted(model["id"] for model in models)
    except (ValueError, KeyError, TypeError):
        print("Unexpected response format.")
        return

    print("\nAvailable model IDs:")
    for model_id in model_ids:
        print(model_id)


if __name__ == "__main__":
    main()