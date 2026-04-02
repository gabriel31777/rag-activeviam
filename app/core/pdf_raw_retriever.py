"""PDF Raw Retriever – direct page-level retrieval from PDF text.

This mode works exclusively with the raw text extracted from PDFs,
without any markdown conversion. It retrieves full pages directly.

Advantages:
  - No loss of information from markdown conversion
  - Preserves original text structure
  - Works well with PDFs containing complex layouts/graphics
  
Use case:
  - PDFs with complex formatting, tables, or graphical elements
  - When markdown conversion quality is poor
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from core.embeddings import embed_query, embed_texts
from core.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFRawRetriever:
    """Retriever that works directly with raw PDF page text."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.vector_store = VectorStore(dataset)
        self.pdf_pages: List[Dict] = []
        self.page_embeddings: List[np.ndarray] = []
        self._build_page_index()

    def _build_page_index(self):
        """Build an index of PDF pages with their embeddings."""
        logger.info("Building PDF raw page index for dataset '%s'", self.dataset)

        # Get all chunks from vector store
        all_chunks = [
            chunk
            for chunk in self.vector_store.get_all_chunks()
            if chunk.get("chunk_type", "content") == "content"
        ]

        if not all_chunks:
            logger.warning("No chunks found in dataset '%s'", self.dataset)
            return

        # Group chunks by (source, page) to reconstruct pages
        pages_dict: Dict[tuple, Dict] = {}
        
        for chunk in all_chunks:
            source = chunk.get("source", "unknown")
            page_num = chunk.get("page")
            
            # Only process chunks with page information (i.e., from PDFs)
            if page_num is None:
                continue
                
            key = (source, page_num)
            
            if key not in pages_dict:
                pages_dict[key] = {
                    "source": source,
                    "page": page_num,
                    "text_parts": [],
                }
            
            pages_dict[key]["text_parts"].append(chunk["text"])

        # Combine chunks into full pages
        for (source, page_num), page_data in pages_dict.items():
            page_text = "\n\n".join(page_data["text_parts"])
            self.pdf_pages.append({
                "source": source,
                "page": page_num,
                "text": page_text,
            })

        if not self.pdf_pages:
            logger.warning("No PDF pages found in dataset '%s'", self.dataset)
            return

        # Create embeddings for each page
        page_texts = [p["text"][:1000] for p in self.pdf_pages]  # Use first 1000 chars for embedding
        embeddings = embed_texts(page_texts)
        self.page_embeddings = [np.array(emb) for emb in embeddings]

        logger.info(
            "PDF raw page index built: %d pages from dataset '%s'",
            len(self.pdf_pages),
            self.dataset,
        )

    def retrieve(self, query: str, top_k: int = 3, max_chars_per_page: int = 2000) -> List[Dict]:
        """Retrieve relevant PDF pages using semantic similarity.

        Args:
            query: User query
            top_k: Number of pages to return (default 3 to limit tokens)
            max_chars_per_page: Maximum characters per page to avoid token limits

        Returns:
            List of page dictionaries with source, page, text, and score
        """
        if not self.pdf_pages or not self.page_embeddings:
            logger.warning("PDF page index not built, returning empty results")
            return []

        # Embed query
        query_embedding = np.array(embed_query(query))

        # Convert page embeddings to matrix
        embeddings_matrix = np.array(self.page_embeddings)

        # Compute similarities
        similarities = np.dot(embeddings_matrix, query_embedding)

        # Get top-k most similar pages
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            page = self.pdf_pages[idx]
            score = float(similarities[idx])
            
            # Truncate page text to avoid token limits
            page_text = page["text"]
            if len(page_text) > max_chars_per_page:
                page_text = page_text[:max_chars_per_page] + "\n\n[...Page truncated to fit token limits...]"

            result = {
                "source": page["source"],
                "page": page["page"],
                "text": page_text,
                "pdf_raw_score": score,
            }
            results.append(result)

        logger.info(
            "PDF raw retrieval: %d pages for query '%s...'",
            len(results),
            query[:50],
        )

        return results
