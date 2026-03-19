"""Configuration module for the RAG application.

All settings are read from environment variables (with sensible defaults).
A .env file is loaded automatically via python-dotenv.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration for the RAG application."""

    # ── LLM ──────────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    # ── Embedding model ──────────────────────────────────────────────
    # BGE-small is retrieval-optimized (384 dim, much better than MiniLM)
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )

    # ── Cross-encoder reranker ───────────────────────────────────────
    reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    use_reranker: bool = field(
        default_factory=lambda: os.getenv("USE_RERANKER", "true").lower()
        == "true"
    )

    # ── Hybrid search (dense + BM25) ────────────────────────────────
    use_hybrid_search: bool = field(
        default_factory=lambda: os.getenv("USE_HYBRID_SEARCH", "true").lower()
        == "true"
    )

    # ── Chunking ─────────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )

    # ── Retrieval ────────────────────────────────────────────────────
    # top_k_initial: how many candidates to retrieve before reranking
    top_k_initial: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_INITIAL", "20"))
    )
    # top_k_rerank: how many to keep after cross-encoder reranking
    top_k_rerank: int = field(
        default_factory=lambda: int(os.getenv("TOP_K_RERANK", "5"))
    )

    # ── Limits ───────────────────────────────────────────────────────
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "8192"))
    )

    # ── Paths (default to ./data for local dev; override in Docker) ─
    documents_path: str = field(
        default_factory=lambda: os.getenv("DOCUMENTS_PATH", "./data/documents")
    )
    vectordb_path: str = field(
        default_factory=lambda: os.getenv("VECTORDB_PATH", "./data/vectordb")
    )


# Singleton
_config = None


def get_config() -> Config:
    """Return a singleton Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
