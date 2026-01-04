#!/usr/bin/env python3
"""List Gemini models that support generateContent for the current API key."""
from __future__ import annotations

import os

from dotenv import load_dotenv

try:
    from google import genai
except ImportError as exc:
    raise SystemExit("google-genai not installed. Install it in the venv.") from exc


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set. Add it to .env and retry.")

    client = genai.Client(api_key=api_key)
    print("Models that support generateContent:")
    found = 0
    for model in client.models.list():
        actions = model.supported_actions or []
        if "generateContent" in actions:
            print(f"- {model.name}")
            found += 1
    if found == 0:
        print("(no models found)")


if __name__ == "__main__":
    main()
