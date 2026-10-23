from pathlib import Path

import requests
from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    config = dotenv_values(root / ".env")

    api_key = config.get("GROQ_API_KEY")
    model = config.get("LLAMA_MODEL") or "llama-3.3-70b-versatile"

    if not api_key:
        print("GROQ_API_KEY is missing from .env.")
        return

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Explain music recommendations using only supplied "
                            "metadata. Describe tags as annotations, not proven "
                            "audio properties. Do not invent lyrics, instruments, "
                            "tempo, or listening history."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Write one short sentence explaining this match: "
                            "the reference song and recommended song both have "
                            "the genre tag 'pop' and mood tag 'romantic'."
                        ),
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 120,
            },
            timeout=45,
        )
    except requests.RequestException:
        print("Network request failed. No credentials were printed.")
        return

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        try:
            error = response.json().get("error", {})
            print("Error type:", error.get("type", "unknown"))
            print("Error code:", error.get("code", "unknown"))
        except ValueError:
            print("The server returned a non-JSON error.")
        return

    try:
        data = response.json()
        explanation = data["choices"][0]["message"]["content"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Empty explanation")
    except (ValueError, KeyError, IndexError, TypeError):
        print("Unexpected response format.")
        return

    print("Model:", data.get("model", model))
    print("Explanation:", explanation.strip())


if __name__ == "__main__":
    main()