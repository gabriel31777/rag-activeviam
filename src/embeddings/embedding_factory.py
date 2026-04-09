"""
embedding_factory.py
Factory pour instancier la bonne fonction d'embedding selon un identifiant.

Utilisation :
    from embeddings import get_embedding_function

    # TF-IDF + SVD (approche de Marie)
    emb = get_embedding_function("tfidf_svd", model_path="/chemin/vers/model.pkl")

    # SentenceTransformers (modèle pré-entraîné)
    emb = get_embedding_function("sentence_transformer")
    emb = get_embedding_function("sentence_transformer", model_name="all-mpnet-base-v2")
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
        Type d'embedding : 'tfidf_svd' ou 'sentence_transformer'.
    model_path : str | Path, optionnel
        Chemin du modèle pickle (requis pour 'tfidf_svd').
    model_name : str, optionnel
        Nom du modèle HuggingFace (pour 'sentence_transformer').

    Retourne
    --------
    Instance compatible avec l'interface ChromaDB EmbeddingFunction.

    Lève
    ----
    ValueError
        Si le type d'embedding n'est pas reconnu.
    """
    embedding_type = embedding_type.lower().strip()

    if embedding_type in ("tfidf_svd", "word2vec", "tfidf", "svd"):
        from .tfidf_svd_embedding import TfidfSvdEmbeddingFunction

        if model_path is None:
            raise ValueError(
                "Le paramètre 'model_path' est requis pour l'embedding TF-IDF + SVD."
            )
        return TfidfSvdEmbeddingFunction(model_path)

    elif embedding_type in ("sentence_transformer", "sentence_transformers", "st", "sbert"):
        from .sentence_transformer_embedding import SentenceTransformerWrapper

        if model_name:
            return SentenceTransformerWrapper(model_name=model_name)
        return SentenceTransformerWrapper()

    else:
        types_valides = ["tfidf_svd", "sentence_transformer"]
        raise ValueError(
            f"Type d'embedding inconnu : '{embedding_type}'. "
            f"Types valides : {types_valides}"
        )
