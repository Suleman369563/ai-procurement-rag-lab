#!/usr/bin/env python3
"""Load documents from disk into plain-text payloads for RAG indexing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    """Simple container for text content and its source path."""

    text: str
    source: str


def _read_text_file(path: Path) -> str:
    # Read UTF-8 text files (txt/md) as-is.
    return path.read_text(encoding="utf-8")


def _read_csv_file(path: Path) -> str:
    # Convert CSV rows into a readable text block.
    lines: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            lines.append(" | ".join(header))
        for row in reader:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def _read_pdf_file(path: Path) -> str:
    # Extract text using pypdf; install it if missing.
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pypdf. Install with `python -m pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _iter_files(data_dir: Path) -> Iterable[Path]:
    # Iterate all files; filtering happens by extension below.
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            yield path


def load_documents(data_dir: Path) -> list[Document]:
    """Load supported documents from the given folder."""
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    # Supported: .txt, .md, .csv, .pdf. Everything else is skipped.
    docs: list[Document] = []
    for path in _iter_files(data_dir):
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            text = _read_text_file(path)
        elif ext == ".csv":
            text = _read_csv_file(path)
        elif ext == ".pdf":
            text = _read_pdf_file(path)
        else:
            continue

        if text.strip():
            docs.append(Document(text=text, source=str(path)))
    return docs
