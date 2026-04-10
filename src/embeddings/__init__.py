"""
Module embeddings — Fonctions d'embedding partagées pour le pipeline RAG.

Fournit :
- TfidfSvdEmbeddingFunction  : embedding basé sur TF-IDF + SVD
- Word2VecEmbeddingFunction  : embedding basé sur un vrai Word2Vec (gensim)
- SentenceTransformerWrapper : wrapper autour de sentence-transformers
- get_embedding_function()   : factory pour instancier l'embedding par nom
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
