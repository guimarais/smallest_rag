# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal RAG (Retrieval-Augmented Generation) system built with [Ollama](https://ollama.com/) for local LLM inference.

## Environment Setup

This project uses Conda for environment management.

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate smallest_rag
```

- Python 3.12
- Ollama must be installed and running separately (not managed via Conda)

## Technology Stack

- **LLM inference:** Ollama (local)
- **Embeddings:** Ollama (`nomic-embed-text`)
- **Vector store:** ChromaDB (persistent, `data/chroma/`)
- **Keyword search:** BM25 (rank-bm25, rebuilt in-memory on start)
- **UI:** Streamlit
- **Language:** Python 3.12
- **Environment:** Conda (`environment.yml`)

## Running the App

```bash
# 1. Install dependencies
conda env create -f environment.yml
conda activate smallest_rag

# 2. Start Ollama (in a separate terminal)
ollama serve

# 3. Launch the app (models are auto-pulled on first run)
streamlit run app.py
```

## Architecture

```
app.py                  # Streamlit UI entry point
config.yaml             # LLM / embedding / retrieval settings
rag/
  config.py             # Dataclasses + YAML loader
  models.py             # Ollama health check + model pull helpers
  ingestion.py          # File loading (TXT/PDF/MD/DOCX) + chunking
  store.py              # ChromaDB + BM25 combined document store
  pipeline.py           # Ingest / retrieve / stream orchestration
data/                   # Auto-created; holds ChromaDB DB (gitignored)
```

Retrieval supports three modes (selectable in sidebar):
- **hybrid** – Reciprocal Rank Fusion over BM25 + cosine results (default)
- **cosine** – Dense vector search via ChromaDB HNSW
- **bm25** – Sparse keyword search
