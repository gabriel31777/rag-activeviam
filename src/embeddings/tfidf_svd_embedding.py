"""
tfidf_svd_embedding.py
Fonction d'embedding basée sur TF-IDF + TruncatedSVD.

Cette classe charge un vectoriseur TF-IDF couplé à une réduction de
dimensionnalité SVD pour produire des vecteurs denses, compatibles avec
l'interface ChromaDB EmbeddingFunction.

Note : Dans la branche originale de Marie, cette classe s'appelait
       'Word2VecEmbeddingFunction'. Le nom a été corrigé pour refléter
       la méthode réelle utilisée (TF-IDF + SVD, pas Word2Vec/gensim).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings


class TfidfSvdEmbeddingFunction(EmbeddingFunction[Documents]):
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
        self._model_path = str(model_path)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.svd = model_data["svd"]
        self.vector_size: int = model_data.get("vector_size", 300)

    def __call__(self, input: Documents) -> Embeddings:
        """Encode une liste de textes en vecteurs denses."""
        if not input:
            return []
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings_array = self.svd.transform(tfidf_vecs).astype(np.float32)
        return [embeddings_array[i] for i in range(embeddings_array.shape[0])]

    @staticmethod
    def name() -> str:
        """Nom de la fonction d'embedding (pour ChromaDB)."""
        return "tfidf_svd"

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration pour la sérialisation."""
        return {"model_path": self._model_path}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "TfidfSvdEmbeddingFunction":
        """Reconstruit la fonction d'embedding depuis la config."""
        return TfidfSvdEmbeddingFunction(model_path=config["model_path"])
