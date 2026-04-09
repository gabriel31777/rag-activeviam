"""
sentence_transformer_embedding.py
Wrapper autour de sentence-transformers, compatible avec l'interface ChromaDB.

Utilise un modèle pré-entraîné (par défaut all-MiniLM-L6-v2) pour encoder
les textes en vecteurs denses de haute qualité sémantique.

Avantage par rapport à TF-IDF + SVD :
- Capture le sens sémantique des phrases (pas seulement les mots-clés)
- Pas besoin d'entraînement sur le corpus cible
- Meilleure généralisation sur des requêtes reformulées

Inconvénient :
- Plus lent (modèle neural)
- Nécessite sentence-transformers + torch
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings


# Modèle par défaut — léger et performant
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerWrapper(EmbeddingFunction[Documents]):
    """Fonction d'embedding compatible ChromaDB utilisant SentenceTransformers.

    Paramètres
    ----------
    model_name : str
        Nom du modèle HuggingFace à utiliser.
        Exemples : 'sentence-transformers/all-MiniLM-L6-v2',
                   'sentence-transformers/all-mpnet-base-v2'
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.vector_size: int = self._model.get_sentence_embedding_dimension()

    def __call__(self, input: Documents) -> Embeddings:
        """Encode une liste de textes en vecteurs denses."""
        if not input:
            return []
        embeddings = self._model.encode(input, show_progress_bar=False)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    @staticmethod
    def name() -> str:
        """Nom de la fonction d'embedding (pour ChromaDB)."""
        return "sentence_transformer_wrapper"

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration pour la sérialisation."""
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "SentenceTransformerWrapper":
        """Reconstruit la fonction d'embedding depuis la config."""
        return SentenceTransformerWrapper(model_name=config["model_name"])
