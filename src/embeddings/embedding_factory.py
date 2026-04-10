"""
embedding_factory.py
Factory pour instancier la bonne fonction d'embedding selon un identifiant.

Utilisation :
    from embeddings import get_embedding_function

    emb = get_embedding_function("tfidf_svd", model_path="model.pkl")
    emb = get_embedding_function("word2vec", model_path="w2v_model.pkl")
    emb = get_embedding_function("sentence_transformer")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def get_embedding_function(
    embedding_type: str,
    model_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
) -> Any:
    """Instancie et retourne une fonction d'embedding selon le type demandé.

    Paramètres
    ----------
    embedding_type : str
        Type d'embedding : 'tfidf_svd', 'word2vec' ou 'sentence_transformer'.
    model_path : str | Path, optionnel
        Chemin du modèle pickle (requis pour 'tfidf_svd' et 'word2vec').
    model_name : str, optionnel
        Nom du modèle HuggingFace (pour 'sentence_transformer').
    """
    embedding_type = embedding_type.lower().strip()

    if embedding_type in ("tfidf_svd", "tfidf", "svd"):
        from .tfidf_svd_embedding import TfidfSvdEmbeddingFunction

        if model_path is None:
            raise ValueError(
                "Le paramètre 'model_path' est requis pour l'embedding TF-IDF + SVD."
            )
        return TfidfSvdEmbeddingFunction(model_path)

    elif embedding_type in ("word2vec", "w2v"):
        from .word2vec_embedding import Word2VecEmbeddingFunction

        if model_path is None:
            raise ValueError(
                "Le paramètre 'model_path' est requis pour l'embedding Word2Vec."
            )
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
            f"Types valides : {types_valides}"
        )
