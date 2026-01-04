#!/usr/bin/env python3
"""Index documents into a local Chroma vector store using Gemini embeddings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.loader import Document, load_documents


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Simple character-based chunking with overlap.
    if chunk_size <= overlap:
        raise SystemExit("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _batch(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    # Yield fixed-size batches for embedding calls.
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _ensure_api_key() -> None:
    # Fail fast if the API key is missing.
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not set. Add it to your .env file.")


def _flatten_documents(
    docs: list[Document],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    # Convert documents into chunk records with metadata for traceability.
    records: list[dict] = []
    for doc_index, doc in enumerate(docs, start=1):
        chunks = _chunk_text(doc.text, chunk_size, overlap)
        for chunk_index, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": f"doc{doc_index}_chunk{chunk_index}",
                    "text": chunk,
                    "metadata": {
                        "source": doc.source,
                        "chunk_index": chunk_index,
                        "doc_index": doc_index,
                    },
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into Chroma.")
    parser.add_argument("--data-dir", default="data", help="Folder with PDFs/CSVs.")
    parser.add_argument("--persist-dir", default="chroma", help="Chroma storage path.")
    parser.add_argument("--collection", default="procurement_docs", help="Collection name.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in chars.")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap in chars.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--reset", action="store_true", help="Reset collection before indexing.")
    args = parser.parse_args()

    # Load GOOGLE_API_KEY from .env if present.
    load_dotenv()
    _ensure_api_key()

    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(data_dir)
    if not docs:
        raise SystemExit(f"No supported documents found in {data_dir}")

    # Flatten text into chunks to keep embeddings small and searchable.
    records = _flatten_documents(docs, args.chunk_size, args.overlap)
    # Gemini embedding model for semantic search.
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    client = chromadb.PersistentClient(path=str(persist_dir))
    # Optional: reset the collection for clean re-indexing.
    if args.reset:
        try:
            client.delete_collection(name=args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)

    print(f"Indexing {len(records)} chunks into '{args.collection}'...")
    for batch in _batch(records, args.batch_size):
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        vectors = embeddings.embed_documents(texts)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)

    print("Indexing complete.")
    print(f"Persisted to: {persist_dir}")


if __name__ == "__main__":
    main()
