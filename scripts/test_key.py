#!/usr/bin/env python3
from __future__ import annotations

import getpass

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def secure_connection_test() -> None:
    """Prompt for an API key and validate access via an embedding call."""
    print("Google API key validation (no key is saved).")

    api_key = getpass.getpass("Google API key: ").strip()
    if not api_key:
        raise SystemExit("Empty input. Aborting.")

    model_name = "text-embedding-004"
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    result = embeddings.embed_query("key validation")
    print("OK")
    print(f"Embedding dimensions: {len(result)}")


if __name__ == "__main__":
    secure_connection_test()
