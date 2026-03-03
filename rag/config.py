from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class RetrieverConfig:
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    search_mode: str = "hybrid"
    bm25_weight: float = 0.5
    cosine_weight: float = 0.5


@dataclass
class Config:
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    retrieval: RetrieverConfig = field(default_factory=RetrieverConfig)


def load_config(path: str = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config()

    with p.open() as f:
        raw = yaml.safe_load(f)

    retrieval_raw = raw.get("retrieval", {})
    hybrid_raw = raw.get("hybrid", {})

    retrieval = RetrieverConfig(
        top_k=retrieval_raw.get("top_k", 5),
        chunk_size=retrieval_raw.get("chunk_size", 500),
        chunk_overlap=retrieval_raw.get("chunk_overlap", 50),
        search_mode=retrieval_raw.get("search_mode", "hybrid"),
        bm25_weight=hybrid_raw.get("bm25_weight", 0.5),
        cosine_weight=hybrid_raw.get("cosine_weight", 0.5),
    )

    return Config(
        llm_model=raw.get("llm", {}).get("model", "llama3.2"),
        embedding_model=raw.get("embedding", {}).get("model", "nomic-embed-text"),
        retrieval=retrieval,
    )
