# smallest_rag

A minimal, fully local RAG (Retrieval-Augmented Generation) system. Upload documents, ask questions, and get answers grounded in your own content — no cloud required.

Built with [Ollama](https://ollama.com/) for local LLM and embedding inference, [ChromaDB](https://www.trychroma.com/) for vector storage, and [Streamlit](https://streamlit.io/) for the UI.

---

## Features

- **Fully local** — LLM inference and embeddings run via Ollama; no data leaves your machine
- **Hybrid search** — combines BM25 keyword search and cosine vector search via Reciprocal Rank Fusion
- **Multiple file formats** — supports `.txt`, `.pdf`, `.md`, and `.docx`
- **Persistent knowledge base** — documents survive app restarts (stored in ChromaDB)
- **Auto model pulling** — missing Ollama models are downloaded automatically on first launch
- **Conversational** — multi-turn chat with full history passed to the LLM

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [Conda](https://docs.conda.io/en/latest/miniconda.html) | Manages the Python environment |
| [Ollama](https://ollama.com/download) | Must be installed and running separately |

---

## Installation

### 1. Install Ollama

Follow the instructions at [ollama.com/download](https://ollama.com/download) for your OS.

### 2. Clone the repository

```bash
git clone https://github.com/your-username/smallest_rag.git
cd smallest_rag
```

### 3. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate smallest_rag
```

This installs all dependencies (Streamlit, ChromaDB, pypdf, python-docx, rank-bm25, ollama, etc.).

---

## Running the App

### 1. Start Ollama

In a separate terminal:

```bash
ollama serve
```

### 2. Launch the Streamlit app

```bash
conda activate smallest_rag
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**On first launch**, the app automatically pulls the required models from Ollama:
- `llama3.2` — LLM for answer generation
- `nomic-embed-text` — embedding model for vector search

This may take a few minutes depending on your internet connection. Progress is shown in the UI.

---

## Usage

### Adding documents

1. Open the **sidebar** on the left.
2. Under **Add document**, click the file uploader and select a `.txt`, `.pdf`, `.md`, or `.docx` file.
3. Click **Add to knowledge base**.
4. The file is chunked and embedded; a progress bar shows the status.
5. The document name appears in the **Knowledge base** list once complete.

### Asking questions

Type a question in the chat input at the bottom of the main area and press Enter.

The app will:
1. Retrieve the most relevant chunks from your knowledge base.
2. Pass them as context to the LLM.
3. Stream the answer back to you.

Each assistant message includes a collapsible **Sources** section showing which document chunks were used.

### Changing search mode

Use the **Search mode** radio buttons in the sidebar to switch between:

| Mode | Description |
|---|---|
| `hybrid` | Combines BM25 + cosine via Reciprocal Rank Fusion (default, usually best) |
| `cosine` | Dense vector search only — good for semantic similarity |
| `bm25` | Sparse keyword search only — good for exact term matching |

### Removing documents

Click the **✕** button next to any document in the **Knowledge base** list to remove it and all its chunks.

### Clearing the conversation

Click **Clear conversation** in the sidebar to reset the chat history. The knowledge base is not affected.

---

## Configuration

Edit `config.yaml` to change default models and retrieval settings:

```yaml
llm:
  model: "llama3.2"          # Any model available in Ollama

embedding:
  model: "nomic-embed-text"  # Any embedding model available in Ollama

retrieval:
  top_k: 5          # Number of chunks retrieved per query
  chunk_size: 500   # Characters per chunk
  chunk_overlap: 50 # Overlap between consecutive chunks
  search_mode: "hybrid"  # Default search mode: hybrid | cosine | bm25

hybrid:
  bm25_weight: 0.5   # Weight for BM25 results in RRF fusion
  cosine_weight: 0.5 # Weight for cosine results in RRF fusion
```

To use a different LLM (e.g. `mistral` or `gemma3`), update `llm.model` and restart the app — it will be pulled automatically if not already present.

---

## Project Structure

```
smallest_rag/
├── app.py            # Streamlit entry point
├── config.yaml       # Model and retrieval configuration
├── environment.yml   # Conda environment spec
├── rag/
│   ├── config.py     # Config dataclasses and YAML loader
│   ├── models.py     # Ollama health check and model pull helpers
│   ├── ingestion.py  # File loading and text chunking
│   ├── store.py      # ChromaDB + BM25 document store
│   └── pipeline.py   # Ingest / retrieve / stream orchestration
└── data/             # Auto-created; holds the ChromaDB database (gitignored)
```

---

## How It Works

1. **Ingestion** — uploaded files are split into overlapping character chunks. Each chunk is embedded using Ollama and stored in ChromaDB alongside its source filename.

2. **Retrieval** — at query time, the question is searched against the knowledge base using the selected mode:
   - *BM25*: tokenised keyword matching scored with Okapi BM25.
   - *Cosine*: the question is embedded and compared to chunk embeddings via HNSW approximate nearest-neighbour search.
   - *Hybrid*: both lists are merged with Reciprocal Rank Fusion (RRF, k=60), weighted by `bm25_weight` and `cosine_weight`.

3. **Generation** — the top-k chunks are injected into a system prompt alongside the full conversation history. The LLM streams its response token by token.

4. **Persistence** — ChromaDB writes to `data/chroma/` automatically. The BM25 index is rebuilt in memory from ChromaDB on each app start.
