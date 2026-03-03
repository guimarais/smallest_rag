from pathlib import Path
from typing import Callable, Generator

import ollama

from .config import Config
from .ingestion import chunk_text, load_document
from .store import DocumentStore


class RAGPipeline:
    def __init__(self, config: Config, data_dir: Path):
        self._config = config
        self._store = DocumentStore(data_dir)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        response = ollama.embeddings(
            model=self._config.embedding_model,
            prompt=text,
        )
        return response.embedding

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        file_path: Path,
        source_name: str,
        progress_cb: Callable[[float], None] | None = None,
    ) -> int:
        text = load_document(file_path)
        cfg = self._config.retrieval
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)

        embeddings = []
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            embeddings.append(self._embed(chunk))
            if progress_cb:
                progress_cb((i + 1) / total)

        self._store.add_chunks(chunks, embeddings, source_name)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, search_mode: str) -> list[dict]:
        cfg = self._config.retrieval
        top_k = cfg.top_k

        if search_mode == "bm25":
            return self._store.bm25_search(question, top_k)

        embedding = self._embed(question)

        if search_mode == "cosine":
            return self._store.cosine_search(embedding, top_k)

        # hybrid (default)
        return self._store.hybrid_search(
            question,
            embedding,
            top_k,
            w_bm25=cfg.bm25_weight,
            w_cos=cfg.cosine_weight,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def stream_response(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> Generator[str, None, None]:
        context = "\n\n---\n\n".join(c["text"] for c in chunks)

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using only "
            "the context provided below. If the context does not contain enough "
            "information, say so.\n\n"
            f"Context:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})

        for chunk in ollama.chat(
            model=self._config.llm_model,
            messages=messages,
            stream=True,
        ):
            yield chunk.message.content

    # ------------------------------------------------------------------
    # Store pass-throughs
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        return self._store.list_sources()

    def delete_source(self, source: str) -> None:
        self._store.delete_source(source)
