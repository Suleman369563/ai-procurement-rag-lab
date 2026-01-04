#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Keep subprocesses anchored to the project root.
REPO_ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    # Run subcommands and bubble up failures.
    print(f"Working dir: {REPO_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _cmd_detect_tables(args: argparse.Namespace) -> None:
    print(f"Loading image: {args.image}")
    print(f"Output dir: {args.out_dir}")
    # Delegate to the OpenCV table detection script.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "cv" / "table_detect.py"),
        "--image",
        args.image,
        "--out-dir",
        args.out_dir,
        "--block-size",
        str(args.block_size),
        "--c-value",
        str(args.c_value),
        "--kernel-scale",
        str(args.kernel_scale),
        "--dilate-iters",
        str(args.dilate_iters),
        "--min-area-ratio",
        str(args.min_area_ratio),
        "--min-width",
        str(args.min_width),
        "--min-height",
        str(args.min_height),
        "--min-aspect",
        str(args.min_aspect),
        "--max-aspect",
        str(args.max_aspect),
    ]
    _run(cmd)


def _cmd_index_docs(args: argparse.Namespace) -> None:
    print(f"Loading documents from: {args.data_dir}")
    print(f"Persist dir: {args.persist_dir}")
    print(f"Collection: {args.collection}")
    # Use the module entry point so imports resolve cleanly.
    cmd = [
        sys.executable,
        "-m",
        "rag.index",
        "--data-dir",
        args.data_dir,
        "--persist-dir",
        args.persist_dir,
        "--collection",
        args.collection,
        "--chunk-size",
        str(args.chunk_size),
        "--overlap",
        str(args.overlap),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.reset:
        cmd.append("--reset")
    _run(cmd)


def _cmd_query(args: argparse.Namespace) -> None:
    print(f"Loading index from: {args.persist_dir}")
    print(f"Collection: {args.collection}")
    # Query the RAG index and print the answer + sources.
    cmd = [
        sys.executable,
        "-m",
        "rag.query",
        "--query",
        args.query,
        "--persist-dir",
        args.persist_dir,
        "--collection",
        args.collection,
        "--top-k",
        str(args.top_k),
    ]
    if args.show_sources:
        cmd.append("--show-sources")
    _run(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Procurement Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Table detection CLI.
    detect = subparsers.add_parser("detect-tables", help="Run OpenCV table detection.")
    detect.add_argument("--image", required=True, help="Path to input image (PNG/JPG).")
    detect.add_argument("--out-dir", default="outputs", help="Output directory.")
    detect.add_argument("--block-size", type=int, default=25, help="Adaptive threshold block size.")
    detect.add_argument("--c-value", type=int, default=15, help="Adaptive threshold C value.")
    detect.add_argument("--kernel-scale", type=int, default=30, help="Kernel scale for line extraction.")
    detect.add_argument("--dilate-iters", type=int, default=2, help="Dilate iterations for line mask.")
    detect.add_argument("--min-area-ratio", type=float, default=0.01, help="Min area ratio.")
    detect.add_argument("--min-width", type=int, default=120, help="Min table width in pixels.")
    detect.add_argument("--min-height", type=int, default=80, help="Min table height in pixels.")
    detect.add_argument("--min-aspect", type=float, default=0.3, help="Min aspect ratio (w/h).")
    detect.add_argument("--max-aspect", type=float, default=6.0, help="Max aspect ratio (w/h).")
    detect.set_defaults(func=_cmd_detect_tables)

    # Indexing CLI.
    index_docs = subparsers.add_parser("index-docs", help="Index documents into Chroma.")
    index_docs.add_argument("--data-dir", default="data", help="Folder with PDFs/CSVs.")
    index_docs.add_argument("--persist-dir", default="chroma", help="Chroma storage path.")
    index_docs.add_argument("--collection", default="procurement_docs", help="Collection name.")
    index_docs.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in chars.")
    index_docs.add_argument("--overlap", type=int, default=150, help="Chunk overlap in chars.")
    index_docs.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    index_docs.add_argument("--reset", action="store_true", help="Reset collection before indexing.")
    index_docs.set_defaults(func=_cmd_index_docs)

    # Query CLI.
    query = subparsers.add_parser("query", help="Query the local RAG index.")
    query.add_argument("--query", required=True, help="User question.")
    query.add_argument("--persist-dir", default="chroma", help="Chroma storage path.")
    query.add_argument("--collection", default="procurement_docs", help="Collection name.")
    query.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    query.add_argument("--show-sources", action="store_true", help="Print source paths.")
    query.set_defaults(func=_cmd_query)

    return parser


def _prompt(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def _interactive_menu() -> None:
    # Simple interactive path for demo use.
    print("Select an action:")
    print("1) Detect tables")
    print("2) Index documents")
    print("3) Query RAG")
    print("q) Quit")

    choice = input("Choice: ").strip().lower()
    if choice in {"q", "quit", "exit"}:
        print("Exiting.")
        return

    if choice == "1":
        image = _prompt("Image path", "data/synthetic_table.png")
        out_dir = _prompt("Output dir", "outputs")
        args = argparse.Namespace(
            image=image,
            out_dir=out_dir,
            block_size=25,
            c_value=15,
            kernel_scale=30,
            dilate_iters=2,
            min_area_ratio=0.01,
            min_width=120,
            min_height=80,
            min_aspect=0.3,
            max_aspect=6.0,
        )
        _cmd_detect_tables(args)
        return

    if choice == "2":
        data_dir = _prompt("Data dir", "data")
        persist_dir = _prompt("Persist dir", "chroma")
        collection = _prompt("Collection", "procurement_docs")
        reset = _prompt("Reset collection? (y/N)", "n").lower().startswith("y")
        args = argparse.Namespace(
            data_dir=data_dir,
            persist_dir=persist_dir,
            collection=collection,
            chunk_size=1000,
            overlap=150,
            batch_size=32,
            reset=reset,
        )
        _cmd_index_docs(args)
        return

    if choice == "3":
        query = _prompt("Query", "Which vendor appears most often in the sample data?")
        persist_dir = _prompt("Persist dir", "chroma")
        collection = _prompt("Collection", "procurement_docs")
        top_k = int(_prompt("Top-k", "4"))
        show_sources = _prompt("Show sources? (Y/n)", "y").lower() != "n"
        args = argparse.Namespace(
            query=query,
            persist_dir=persist_dir,
            collection=collection,
            top_k=top_k,
            show_sources=show_sources,
        )
        _cmd_query(args)
        return

    print("Invalid choice.")


def main() -> None:
    if len(sys.argv) == 1:
        _interactive_menu()
        return

    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
