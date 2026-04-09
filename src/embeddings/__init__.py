"""
Module embeddings — Fonctions d'embedding partagées pour le pipeline RAG.

Fournit :
- TfidfSvdEmbeddingFunction  : embedding basé sur TF-IDF + SVD (approche de Marie)
- SentenceTransformerWrapper  : wrapper autour de sentence-transformers (all-MiniLM-L6-v2, etc.)
- get_embedding_function()    : factory pour instancier l'embedding par nom
"""

from .tfidf_svd_embedding import TfidfSvdEmbeddingFunction
from .sentence_transformer_embedding import SentenceTransformerWrapper
from .embedding_factory import get_embedding_function

# Alias rétro-compatible
Word2VecEmbeddingFunction = TfidfSvdEmbeddingFunction

__all__ = [
    "TfidfSvdEmbeddingFunction",
    "Word2VecEmbeddingFunction",
    "SentenceTransformerWrapper",
    "get_embedding_function",
]
