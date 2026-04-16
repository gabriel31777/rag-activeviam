"""
Module embeddings -- fonctions d'embedding pour le pipeline RAG.

Expose :
- TfidfSvdEmbeddingFunction
- Word2VecEmbeddingFunction
- SentenceTransformerWrapper
- get_embedding_function() (factory)
"""

from .tfidf_svd_embedding import TfidfSvdEmbeddingFunction
from .word2vec_embedding import Word2VecEmbeddingFunction
from .sentence_transformer_embedding import SentenceTransformerWrapper
from .embedding_factory import get_embedding_function

__all__ = [
    "TfidfSvdEmbeddingFunction",
    "Word2VecEmbeddingFunction",
    "SentenceTransformerWrapper",
    "get_embedding_function",
]
