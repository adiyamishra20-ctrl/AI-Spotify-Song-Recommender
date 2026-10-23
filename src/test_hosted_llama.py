from pathlib import Path

import requests
from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    config = dotenv_values(root / ".env")
    key = config.get("OPENROUTER_API_KEY")
    model = config.get("OPENROUTER_MODEL")

    if not key or not model:
        print("Missing OPENROUTER_API_KEY or OPENROUTER_MODEL in .env.")
        return

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Use only supplied facts. Write one short sentence. "
                            "Do not infer tempo, lyrics, or instruments."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Explain that two songs share the listener tags "
                            "'pop' and 'romantic'."
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=(10, 60),
        )
    except requests.RequestException:
        print("Network request failed.")
        return

    print("HTTP status:", response.status_code)

    try:
        data = response.json()
    except ValueError:
        print("Response was not JSON.")
        return

    if response.status_code != 200 or "error" in data:
        error = data.get("error", {})
        print("API error code:", error.get("code", "unknown"))
        print("Hosted access is not confirmed.")
        return

    try:
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response")
    except (KeyError, IndexError, TypeError, ValueError):
        print("No usable explanation returned.")
        return

    print("Model:", data.get("model", model))
    print("Explanation:", text.strip())


if __name__ == "__main__":
    main()