import hashlib
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi


def _sha_prefix(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:16]


class DocumentStore:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []  # [{id, text, source}]
        self._rebuild_bm25()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result["ids"]
        docs = result["documents"] or []
        metas = result["metadatas"] or []

        self._bm25_docs = [
            {"id": id_, "text": text, "source": (meta or {}).get("source", "")}
            for id_, text, meta in zip(ids, docs, metas)
        ]

        if self._bm25_docs:
            tokenized = [d["text"].lower().split() for d in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source: str,
    ) -> None:
        # Remove any existing chunks for this source first
        self.delete_source(source)

        prefix = _sha_prefix(source)
        ids = [f"{prefix}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._rebuild_bm25()

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})
        self._rebuild_bm25()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        metas = self._collection.get(include=["metadatas"])["metadatas"] or []
        sources = sorted({(m or {}).get("source", "") for m in metas} - {""})
        return sources

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(scores, self._bm25_docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {"id": d["id"], "text": d["text"], "source": d["source"], "score": float(s)}
            for s, d in ranked[:top_k]
            if s > 0
        ]

    def cosine_search(self, embedding: list[float], top_k: int) -> list[dict]:
        if not self._bm25_docs:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, len(self._bm25_docs)),
            include=["documents", "metadatas", "distances"],
        )
        docs = result["documents"][0] or []
        metas = result["metadatas"][0] or []
        distances = result["distances"][0] or []
        ids = result["ids"][0] or []

        return [
            {
                "id": id_,
                "text": text,
                "source": (meta or {}).get("source", ""),
                "score": float(1 - dist),
            }
            for id_, text, meta, dist in zip(ids, docs, metas, distances)
        ]

    def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int,
        w_bm25: float = 0.5,
        w_cos: float = 0.5,
    ) -> list[dict]:
        """Reciprocal Rank Fusion with per-method weights."""
        k = 60  # RRF constant

        bm25_results = self.bm25_search(query, top_k * 2)
        cos_results = self.cosine_search(embedding, top_k * 2)

        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        sources: dict[str, str] = {}

        for rank, hit in enumerate(bm25_results):
            id_ = hit["id"]
            scores[id_] = scores.get(id_, 0.0) + w_bm25 / (k + rank + 1)
            texts[id_] = hit["text"]
            sources[id_] = hit["source"]

        for rank, hit in enumerate(cos_results):
            id_ = hit["id"]
            scores[id_] = scores.get(id_, 0.0) + w_cos / (k + rank + 1)
            texts[id_] = hit["text"]
            sources[id_] = hit["source"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"id": id_, "text": texts[id_], "source": sources[id_], "score": score}
            for id_, score in ranked[:top_k]
        ]
