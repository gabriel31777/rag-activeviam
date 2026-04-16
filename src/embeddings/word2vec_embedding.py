"""
word2vec_embedding.py
Embedding Word2Vec (gensim) compatible ChromaDB.
Moyenne ponderee TF-IDF des vecteurs de mots.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings


class Word2VecEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding ChromaDB via Word2Vec (gensim)."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = str(model_path)
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.w2v_model = model_data["w2v_model"]
        self.vectorizer = model_data["vectorizer"]
        self.vector_size: int = model_data.get("vector_size", 300)

        # Cache du vocabulaire TF-IDF
        self._tfidf_vocab = self.vectorizer.vocabulary_
        self._idf = dict(
            zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_)
        )

    def _embed_one(self, text: str) -> np.ndarray:
        """Encode un texte via moyenne ponderee TF-IDF des vecteurs de mots."""
        tokens = text.lower().split()
        vec = np.zeros(self.vector_size, dtype=np.float32)
        weight_sum = 0.0

        for token in tokens:
            if token in self.w2v_model.wv and token in self._idf:
                w = self._idf[token]
                vec += w * self.w2v_model.wv[token]
                weight_sum += w

        if weight_sum > 0:
            vec /= weight_sum

        return vec

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        return [self._embed_one(text) for text in input]

    @staticmethod
    def name() -> str:
        return "word2vec_gensim"

    def get_config(self) -> Dict[str, Any]:
        return {"model_path": self._model_path}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "Word2VecEmbeddingFunction":
        return Word2VecEmbeddingFunction(model_path=config["model_path"])
