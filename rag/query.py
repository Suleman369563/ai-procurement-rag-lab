#!/usr/bin/env python3
"""Query the local Chroma store and answer using Gemini 1.5 Pro."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def _ensure_api_key() -> None:
    # Fail fast if the API key is missing.
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set. Add it to your .env file.")


def _build_prompt(question: str, contexts: list[str]) -> str:
    # Simple RAG prompt that forces grounding in retrieved context.
    context_block = "\n\n".join(contexts) if contexts else "No context available."
    return (
        "You are a procurement analyst. Answer using only the provided context. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _extract_answer(response) -> str:
    # Handle LangChain response objects consistently.
    return getattr(response, "content", str(response)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("--query", required=True, help="User question.")
    parser.add_argument("--persist-dir", default="chroma", help="Chroma storage path.")
    parser.add_argument("--collection", default="procurement_docs", help="Collection name.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--show-sources", action="store_true", help="Print source paths.")
    args = parser.parse_args()

    # Load GOOGLE_API_KEY from .env if present.
    load_dotenv()
    _ensure_api_key()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise SystemExit(f"Chroma directory not found: {persist_dir}")

    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=args.collection)

    # Convert the question into a vector for similarity search.
    query_vector = embeddings.embed_query(args.query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    # Build the context window from retrieved documents.
    contexts = [doc for doc in documents if doc]

    prompt = _build_prompt(args.query, contexts)
    # Ask the model to answer using only the retrieved context.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    response = llm.invoke(prompt)
    answer = _extract_answer(response)

    print("\nAnswer:\n")
    print(answer)

    if args.show_sources:
        print("\nSources:")
        for meta in metadatas:
            print(f"- {meta.get('source')}")


if __name__ == "__main__":
    main()
