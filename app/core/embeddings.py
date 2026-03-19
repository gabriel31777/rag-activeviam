"""Embedding generation using sentence-transformers.

Uses a bi-encoder model optimized for retrieval tasks.
Default: BAAI/bge-small-en-v1.5 (384-dim, significantly better than
all-MiniLM-L6-v2 on retrieval benchmarks).

Embeddings are L2-normalized so that cosine similarity reduces to a
simple dot product, which ChromaDB handles efficiently.
"""

from __future__ import annotations

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Module-level singleton
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", config.embedding_model)
        _model = SentenceTransformer(config.embedding_model)
        dim = _model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d).", dim)
    return _model


def embed_texts(
    texts: List[str], batch_size: int = 64
) -> List[List[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: Texts to embed.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors (L2-normalized).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string (L2-normalized)."""
    model = _get_model()
    embedding = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding.tolist()
