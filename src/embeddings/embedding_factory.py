"""
embedding_factory.py
Factory pour instancier la fonction d'embedding par identifiant.

Utilisation :
    from embeddings import get_embedding_function
    emb = get_embedding_function("tfidf_svd", model_path="model.pkl")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def get_embedding_function(
    embedding_type: str,
    model_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
) -> Any:
    """Retourne la fonction d'embedding correspondant au type demande.

    Args:
        embedding_type : 'tfidf_svd', 'word2vec' ou 'sentence_transformer'
        model_path : chemin du pickle (requis pour tfidf_svd et word2vec)
        model_name : nom HuggingFace (pour sentence_transformer)
    """
    embedding_type = embedding_type.lower().strip()

    if embedding_type in ("tfidf_svd", "tfidf", "svd"):
        from .tfidf_svd_embedding import TfidfSvdEmbeddingFunction

        if model_path is None:
            raise ValueError("model_path requis pour l'embedding TF-IDF + SVD.")
        return TfidfSvdEmbeddingFunction(model_path)

    elif embedding_type in ("word2vec", "w2v"):
        from .word2vec_embedding import Word2VecEmbeddingFunction

            raise ValueError("model_path requis pour l'embedding Word2Vec.")
        return Word2VecEmbeddingFunction(model_path)

    elif embedding_type in ("sentence_transformer", "sentence_transformers", "st", "sbert"):
        from .sentence_transformer_embedding import SentenceTransformerWrapper

        if model_name:
            return SentenceTransformerWrapper(model_name=model_name)
        return SentenceTransformerWrapper()

    else:
        types_valides = ["tfidf_svd", "word2vec", "sentence_transformer"]
        raise ValueError(
            f"Type d'embedding inconnu : '{embedding_type}'. "
            f"Valides : {types_valides}"
        )
