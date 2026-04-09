"""
tfidf_svd_embedding.py
Fonction d'embedding basée sur TF-IDF + TruncatedSVD.

Cette classe entraîne (ou charge) un vectoriseur TF-IDF couplé à une
réduction de dimensionnalité SVD pour produire des vecteurs denses.
Compatible avec l'interface ChromaDB EmbeddingFunction.

Note : Dans la branche originale de Marie, cette classe s'appelait
       'Word2VecEmbeddingFunction'. Le nom a été corrigé pour refléter
       la méthode réelle utilisée (TF-IDF + SVD, pas Word2Vec/gensim).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List


class TfidfSvdEmbeddingFunction:
    """Fonction d'embedding compatible ChromaDB utilisant TF-IDF + SVD.

    Paramètres
    ----------
    model_path : str | Path
        Chemin vers le fichier pickle contenant le modèle entraîné.
        Le fichier doit contenir un dict avec les clés :
        - 'vectorizer' : TfidfVectorizer entraîné
        - 'svd'        : TruncatedSVD entraîné
        - 'vector_size': int (dimension des vecteurs, par défaut 300)
    """

    def __init__(self, model_path: str | Path) -> None:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.svd = model_data["svd"]
        self.vector_size: int = model_data.get("vector_size", 300)

    # -- Interface ChromaDB --------------------------------------------------

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Encode une liste de textes en vecteurs denses."""
        if not input:
            return []
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings_array = self.svd.transform(tfidf_vecs)
        return [list(row) for row in embeddings_array]

    def embed_query(self, input: str) -> List[float]:
        """Encode une seule requête en vecteur dense."""
        if isinstance(input, list):
            return self(input)[0]
        return self([input])[0]

    def name(self) -> str:
        """Nom de la fonction d'embedding (pour ChromaDB)."""
        return "tfidf_svd"
