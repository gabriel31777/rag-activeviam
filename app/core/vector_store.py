"""ChromaDB vector store wrapper.

Provides dataset-scoped vector storage with:
- Pre-computed embeddings (from the embeddings module)
- Cosine similarity search (HNSW index)
- Metadata filtering (source, page)
- Full collection retrieval (for BM25 index building)
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class VectorStore:
    """Wrapper around ChromaDB for dataset-scoped vector storage."""

    def __init__(self, dataset: str, persist_dir: Optional[str] = None):
        self.dataset = dataset
        self.persist_dir = persist_dir or os.path.join(
            config.vectordb_path, dataset
        )
        os.makedirs(self.persist_dir, exist_ok=True)

        try:
            self._init_client()
        except Exception as e:
            # ChromaDB corruption (e.g. "no such table: acquire_write")
            # → wipe the directory and recreate from scratch
            logger.warning(
                "ChromaDB init failed for '%s' (%s). "
                "Wiping corrupt DB and recreating.",
                dataset,
                e,
            )
            if os.path.isdir(self.persist_dir):
                shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
            self._init_client()

        logger.info(
            "VectorStore ready: '%s' at %s (%d docs)",
            dataset,
            self.persist_dir,
            self.collection.count(),
        )

    def _init_client(self) -> None:
        """Initialize the ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.dataset,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
    ) -> None:
        """Add chunks with precomputed embeddings.

        The `text` field of each chunk is stored as the document.
        Embeddings are precomputed from `text_to_embed` (contextual text).
        """
        if not chunks:
            return

        # ChromaDB batch limit: process in batches of 5000
        batch_size = 5000
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [
                {
                    "dataset": c["dataset"],
                    "source": c["source"],
                    "page": c["page"] if c["page"] is not None else -1,
                }
                for c in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(
            "Added %d chunks to dataset '%s'", len(chunks), self.dataset
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Perform cosine similarity search.

        Returns list of dicts with: chunk_id, text, source, page, score.
        Lower score = more similar (cosine distance).
        """
        k = top_k or config.top_k_initial
        count = self.collection.count()
        if count == 0:
            return []

        k = min(k, count)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            page = meta.get("page", -1)
            output.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "source": meta.get("source", ""),
                    "page": page if page != -1 else None,
                    "score": results["distances"][0][i],
                }
            )
        return output

    def get_all_chunks(self) -> List[Dict]:
        """Retrieve ALL chunks from the collection.

        Used for building the BM25 index at query time.
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        output: List[Dict] = []
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i]
            page = meta.get("page", -1)
            output.append(
                {
                    "chunk_id": results["ids"][i],
                    "text": results["documents"][i],
                    "source": meta.get("source", ""),
                    "page": page if page != -1 else None,
                }
            )
        return output

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (full rebuild)."""
        self.client.delete_collection(self.dataset)
        self.collection = self.client.get_or_create_collection(
            name=self.dataset,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset vector store for '%s'", self.dataset)
