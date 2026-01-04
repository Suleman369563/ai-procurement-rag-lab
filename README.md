# Project 2: AI Procurement Agent

Build a local RAG pipeline for procurement documents, plus OpenCV-based table detection.

## Features
- Synthetic procurement data generation (CSV, PNG, PDF).
- Table detection with debug overlays for visual QA.
- RAG indexing with Gemini embeddings and ChromaDB.
- CLI and n8n workflow for orchestration.

## Layout
- `data/`: synthetic CSV/PNG/PDF inputs.
- `cv/`: table detection module.
- `rag/`: RAG loader/index/query modules.
- `outputs/`: debug images and table detection outputs.
- `chroma/`: local vector store (generated).
- `main.py`: CLI entry point.
- `n8n/`: workflow and runbook.

## Setup
1) Create a virtualenv:
   ```bash
   cd /path/to/project-2-ai-procurement-agent
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Add your API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set `GOOGLE_API_KEY`.

## Run locally
Generate synthetic data:
```bash
python data/generate_synthetic.py --out-dir data
```

Detect tables:
```bash
python main.py detect-tables --image data/synthetic_table.png --out-dir outputs
```

Index documents:
```bash
python main.py index-docs --data-dir data --persist-dir chroma --collection procurement_docs
```

Query the index:
```bash
python main.py query \
  --query "Which vendor appears most often in the sample data?" \
  --persist-dir chroma \
  --collection procurement_docs \
  --show-sources
```

## n8n Orchestration
See `n8n/README.md` for the Docker + SSH workflow.
