"""
tfidf_svd_embedding.py
Embedding TF-IDF + TruncatedSVD compatible ChromaDB.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings


class TfidfSvdEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding ChromaDB via TF-IDF + SVD."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = str(model_path)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.svd = model_data["svd"]
        self.vector_size: int = model_data.get("vector_size", 300)

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings_array = self.svd.transform(tfidf_vecs).astype(np.float32)
        return [embeddings_array[i] for i in range(embeddings_array.shape[0])]

    @staticmethod
    def name() -> str:
        return "tfidf_svd"

    def get_config(self) -> Dict[str, Any]:
        return {"model_path": self._model_path}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "TfidfSvdEmbeddingFunction":
        return TfidfSvdEmbeddingFunction(model_path=config["model_path"])
