"""
sentence_transformer_embedding.py
Wrapper SentenceTransformers compatible avec ChromaDB.
Utilise un modele pre-entraine (par defaut all-MiniLM-L6-v2).
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings


# Modele par defaut
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerWrapper(EmbeddingFunction[Documents]):
    """Embedding ChromaDB via SentenceTransformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.vector_size: int = self._model.get_sentence_embedding_dimension()

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        embeddings = self._model.encode(input, show_progress_bar=False)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    @staticmethod
    def name() -> str:
        return "sentence_transformer_wrapper"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "SentenceTransformerWrapper":
        return SentenceTransformerWrapper(model_name=config["model_name"])
