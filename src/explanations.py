import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "processed" / "llama_explanations.json"


def explain_recommendation(reference, candidate, selected_mood=None):
    shared_genres = sorted(
        set(reference["genres"]) & set(candidate["genres"])
    )
    shared_moods = sorted(
        set(reference["moods"]) & set(candidate["moods"])
    )

    evidence = {
        "reference_title": reference["title"],
        "recommended_title": candidate["title"],
        "shared_genre_tags": shared_genres,
        "shared_mood_tags": shared_moods,
        "requested_mood": selected_mood,
        "requested_mood_is_tagged": (
            selected_mood in candidate["moods"]
            if selected_mood else False
        ),
    }

    # Include all supplied inputs so changed annotations invalidate the cache.
    payload = {
        "version": 1,
        "reference": reference,
        "candidate": candidate,
        "selected_mood": selected_mood,
    }
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=False
    )
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            raise ValueError("Explanation cache must be a JSON object.")

    saved = cache.get(key)
    if (
        isinstance(saved, dict)
        and saved.get("evidence") == evidence
        and saved.get("text")
    ):
        return {
            "text": saved["text"],
            "source": "Pre-generated Llama 3.2 explanation",
            "evidence": evidence,
        }

    # Enable this ONLY on your laptop while preparing saved explanations.
    # Cloud deployment leaves it disabled and never contacts Ollama.
    if os.environ.get("SONG_COMPASS_BUILD_CACHE") == "1":
        from src.local_explanations import (
            explain_recommendation as generate_locally,
        )

        result = generate_locally(reference, candidate, selected_mood)

        if result["source"] == "Llama 3.2":
            cache[key] = {
                "text": result["text"],
                "evidence": evidence,
                "model": "llama3.2:3b",
            }
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

            temporary = CACHE_PATH.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(CACHE_PATH)

            return {
                "text": result["text"],
                "source": "Pre-generated Llama 3.2 explanation",
                "evidence": evidence,
            }

    facts = []
    if shared_genres:
        facts.append("Shared listener genre tags: " + ", ".join(shared_genres))
    if shared_moods:
        facts.append("Shared listener mood tags: " + ", ".join(shared_moods))
    if evidence["requested_mood_is_tagged"]:
        facts.append(
            f"The recommendation has the requested mood tag: {selected_mood}"
        )

    return {
        "text": (
            ". ".join(facts) + "."
            if facts
            else "No shared genre or mood tags are available for this pair."
        ),
        "source": "Template fallback",
        "evidence": evidence,
    }