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

from typing import List


# Modèle par défaut — léger et performant
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerWrapper:
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

    # -- Interface ChromaDB --------------------------------------------------

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Encode une liste de textes en vecteurs denses."""
        if not input:
            return []
        embeddings = self._model.encode(input, show_progress_bar=False)
        return [list(row) for row in embeddings]

    def embed_query(self, input: str) -> List[float]:
        """Encode une seule requête en vecteur dense."""
        if isinstance(input, list):
            return self(input)[0]
        return self([input])[0]

    def name(self) -> str:
        """Nom de la fonction d'embedding (pour ChromaDB)."""
        return f"sentence_transformer_{self.model_name.split('/')[-1]}"
