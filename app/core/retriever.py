"""Advanced retriever with hybrid search and cross-encoder reranking.

Pipeline (inspired by state-of-the-art RAG systems like NotebookLM):

1. **Query decomposition** – LLM generates sub-queries for maximum recall.
2. **Dense search** – Embedding similarity via ChromaDB (per sub-query).
3. **Sparse search** – BM25 keyword matching (per sub-query).
4. **Reciprocal Rank Fusion** – Merge dense + sparse results.
5. **Cross-encoder reranking** – Second-pass precision scoring.
6. **Return top-k** most relevant chunks.

The cross-encoder reranking step is the single biggest accuracy improvement.
Unlike bi-encoders (which encode query and document independently), a
cross-encoder processes the (query, document) pair together, allowing it
to model fine-grained relevance interactions.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.embeddings import embed_query
from core.vector_store import VectorStore
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# ---------------------------------------------------------------------------
# Singleton for the cross-encoder reranker (heavy to load)
# ---------------------------------------------------------------------------
_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranking model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker: %s", config.reranker_model)
        _reranker = CrossEncoder(config.reranker_model)
        logger.info("Reranker loaded.")
    return _reranker


# ---------------------------------------------------------------------------
# Query decomposition prompt
# ---------------------------------------------------------------------------
_DECOMPOSE_PROMPT = """\
You are a search query optimizer for a document retrieval system.

Given the user's question, generate alternative search queries to maximize recall.

Produce a JSON object with:
1. "sub_queries": a list of 3-5 alternative phrasings of the question. \
Include the original question. Each phrasing should emphasize different \
keywords, synonyms, or angles.
2. "keywords": a list of 3-8 important keywords or named entities from \
the question.

Respond ONLY with valid JSON. No markdown fences. No explanation.

User question: {question}
"""


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------


class Retriever:
    """Hybrid retriever: multi-query + dense + BM25 + cross-encoder rerank."""

    def __init__(self, dataset: str, llm=None):
        self.dataset = dataset
        self.vector_store = VectorStore(dataset)
        self.llm = llm
        # BM25 index (built lazily on first sparse search)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_chunks: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict]:
        """Main retrieval pipeline.

        Steps:
            1. Generate sub-queries (if LLM available)
            2. For each sub-query: dense + BM25 → RRF merge
            3. Deduplicate across sub-queries
            4. Cross-encoder rerank
            5. Filter low-relevance chunks
            6. Consolidate by document (if one doc dominates, keep only it)
            7. Return useful sources only

        Args:
            query: User question.
            top_k: Final number of chunks to return.

        Returns:
            List of chunk dicts, sorted by relevance.
        """
        k_initial = config.top_k_initial
        k_final = top_k or config.top_k_rerank

        # Step 1: Generate sub-queries
        if self.llm:
            sub_queries = self._decompose_question(query)
        else:
            sub_queries = [query]

        logger.info("Retrieval: %d sub-queries", len(sub_queries))

        # Step 2: Multi-query retrieval with hybrid fusion
        all_candidates: Dict[str, Dict] = {}

        for sq in sub_queries:
            candidates = self._hybrid_search(sq, top_k=k_initial)
            for c in candidates:
                cid = c["chunk_id"]
                if cid not in all_candidates or c.get("rrf_score", 0) > all_candidates[cid].get("rrf_score", 0):
                    all_candidates[cid] = c

        candidates_list = list(all_candidates.values())
        logger.info("Unique candidates after multi-query: %d", len(candidates_list))

        if not candidates_list:
            return []

        # Step 3: Cross-encoder reranking
        if config.use_reranker and len(candidates_list) > 1:
            candidates_list = self._rerank(
                query, candidates_list, top_k=max(k_final, 10)
            )
        else:
            # Fallback: sort by RRF score
            candidates_list.sort(
                key=lambda x: x.get("rrf_score", 0), reverse=True
            )
            candidates_list = candidates_list[:max(k_final, 10)]

        # Step 4: Filter and consolidate sources
        candidates_list = self._filter_and_consolidate(
            candidates_list, k_final
        )

        return candidates_list

    # ------------------------------------------------------------------
    # Smart source filtering & consolidation
    # ------------------------------------------------------------------

    def _filter_and_consolidate(
        self,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """Filter out low-relevance chunks and consolidate by document.

        Logic:
        1. Drop chunks whose rerank_score is too far below the best
           (relative threshold: < 30% of top score, or negative when
           top is positive).
        2. If all remaining chunks come from the same document,
           deduplicate by page (keep the best chunk per page).
        3. If one document dominates (≥80% of remaining chunks),
           keep only that document's chunks.
        4. Otherwise return top_k best chunks as-is.
        """
        if not candidates:
            return []

        # --- 1. Relevance threshold filtering ---
        top_score = candidates[0].get("rerank_score")
        if top_score is not None:
            # Dynamic threshold: keep chunks that are reasonably relevant
            if top_score > 0:
                threshold = top_score * 0.25  # within 25% of best
            else:
                threshold = top_score - 2.0   # allow 2-point margin

            filtered = [
                c for c in candidates
                if c.get("rerank_score", float("-inf")) >= threshold
            ]
            # Always keep at least 1
            if not filtered:
                filtered = candidates[:1]
        else:
            filtered = candidates

        logger.info(
            "Relevance filter: %d → %d candidates",
            len(candidates),
            len(filtered),
        )

        # --- 2. Consolidate by document ---
        # Count how many chunks per source
        source_counts: Dict[str, int] = {}
        for c in filtered:
            src = c["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

        total = len(filtered)
        dominant_source = max(source_counts, key=source_counts.get)
        dominant_ratio = source_counts[dominant_source] / total

        if len(source_counts) == 1 or dominant_ratio >= 0.8:
            # One document dominates — keep only its chunks, dedup by page
            doc_chunks = [
                c for c in filtered if c["source"] == dominant_source
            ]
            # Deduplicate by page: keep best-scoring chunk per page
            # For non-PDF docs (page=None), use chunk_id to avoid
            # collapsing all chunks into one.
            seen_pages = set()
            deduped = []
            for c in doc_chunks:  # already sorted by score
                page_key = c.get("page")
                if page_key is None:
                    # Non-PDF: no page info, keep all chunks
                    deduped.append(c)
                elif page_key not in seen_pages:
                    seen_pages.add(page_key)
                    deduped.append(c)

            logger.info(
                "Source consolidation: single doc '%s' → %d unique pages",
                dominant_source,
                len(deduped),
            )
            return deduped[:top_k]

        # --- 3. Multiple relevant documents — return top_k ---
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Hybrid search (dense + BM25)
    # ------------------------------------------------------------------

    def _hybrid_search(
        self, query: str, top_k: int
    ) -> List[Dict]:
        """Combine dense vector search and BM25 sparse search with RRF."""

        # Dense search
        query_embedding = embed_query(query)
        dense_results = self.vector_store.query(
            query_embedding, top_k=top_k
        )

        # Sparse search (BM25)
        if config.use_hybrid_search:
            sparse_results = self._bm25_search(query, top_k=top_k)
        else:
            sparse_results = []

        # Merge with Reciprocal Rank Fusion
        if sparse_results:
            merged = self._reciprocal_rank_fusion(
                dense_results, sparse_results
            )
        else:
            # Pure dense: assign rank-based scores
            for i, r in enumerate(dense_results):
                r["rrf_score"] = 1.0 / (60 + i + 1)
            merged = dense_results

        return merged

    # ------------------------------------------------------------------
    # BM25 sparse search
    # ------------------------------------------------------------------

    def _bm25_search(
        self, query: str, top_k: int
    ) -> List[Dict]:
        """Perform BM25 keyword search over all indexed chunks."""
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None or not self._bm25_chunks:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self._bm25_chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)

        return results

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all chunks in the vector store."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            return

        self._bm25_chunks = all_chunks
        tokenized_corpus = [_tokenize(c["text"]) for c in all_chunks]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index: %d documents", len(all_chunks))

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Merge dense and sparse results using RRF.

        RRF score = sum(1 / (k + rank)) across result lists.
        """
        chunk_map: Dict[str, Dict] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = result.copy()

        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = result.copy()

        for cid in chunk_map:
            chunk_map[cid]["rrf_score"] = rrf_scores[cid]

        merged = sorted(
            chunk_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        return merged

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """Rerank candidates using a cross-encoder for precision.

        The cross-encoder scores each (query, document) pair jointly,
        providing much more accurate relevance judgments than bi-encoder
        similarity alone.
        """
        reranker = _get_reranker()

        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "Reranked %d candidates → top score: %.4f, cutoff: %.4f",
            len(candidates),
            candidates[0]["rerank_score"] if candidates else 0,
            candidates[min(top_k, len(candidates)) - 1]["rerank_score"]
            if candidates
            else 0,
        )

        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Query decomposition
    # ------------------------------------------------------------------

    def _decompose_question(self, question: str) -> List[str]:
        """Use the LLM to generate search sub-queries for better recall."""
        prompt = _DECOMPOSE_PROMPT.format(question=question)

        try:
            raw = self.llm.generate(prompt)
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            data = json.loads(raw)

            sub_queries: List[str] = data.get("sub_queries", [question])
            keywords: List[str] = data.get("keywords", [])

            # Always include the original question
            if question not in sub_queries:
                sub_queries.insert(0, question)

            # Add a keyword-only query for broad recall
            if keywords:
                kw_query = " ".join(keywords)
                if kw_query not in sub_queries:
                    sub_queries.append(kw_query)

            logger.info(
                "Decomposed into %d sub-queries: %s",
                len(sub_queries),
                sub_queries,
            )
            return sub_queries

        except Exception as e:
            logger.warning(
                "Query decomposition failed (%s), using original query", e
            )
            return [question]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple word tokenization with lowercasing for BM25."""
    return re.findall(r"\w+", text.lower())
