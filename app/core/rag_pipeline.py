"""RAG pipeline – orchestrates ingestion and query answering.

Two flows:
  Ingestion: Load → Chunk (contextual) → Embed → Store in ChromaDB
  Query:     Decompose → Hybrid Search → Rerank → Prompt LLM → Answer
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.chunking import chunk_document
from core.embeddings import embed_texts
from core.index_metadata import IndexMetadata
from core.page_index_rag import PageIndexRetriever
from core.pdf_raw_retriever import PDFRawRetriever
from core.retriever import Retriever
from core.vector_store import VectorStore
from dataset.loaders import load_documents_from_directory
from dataset.manager import DatasetManager
from llm.gemini_llm import GeminiLLM
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise research assistant. Your purpose is to answer questions \
using ONLY the provided source material. You never use outside knowledge. \
You always cite your sources."""

ANSWER_PROMPT = """\
## Source Material

{context}

---

## Question

{question}

## Instructions

1. Read ALL provided sources carefully before answering.
2. Answer based ONLY on the source material above. Do NOT use outside knowledge.
3. When citing specific facts or data, quote the relevant text using "quotation marks".
4. Reference sources using [Source N] notation (e.g., [Source 1], [Source 2]).
5. If information comes from multiple sources, synthesize clearly and cite each.
6. If the answer is NOT in the sources, say: "This information is not found in the provided sources."
7. For numerical data or statistics, always cite the exact source.
8. Structure your answer with clear paragraphs or bullet points as appropriate.
9. Be precise, thorough, and well-organized.

Answer:"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline: ingestion + query."""

    def __init__(self):
        self.manager = DatasetManager()
        self.llm = GeminiLLM()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_dataset(
        self,
        dataset: str,
        progress_callback=None,
    ) -> int:
        """Ingest (or re-ingest) all documents in a dataset.

        Args:
            dataset: Dataset name.
            progress_callback: Optional callable(step, total, message)
                called after each document is processed.

        Returns the number of chunks inserted.
        """
        dataset_path = self.manager.get_dataset_path(dataset)
        documents = load_documents_from_directory(dataset_path)

        if not documents:
            logger.warning("No documents in dataset '%s'", dataset)
            return 0

        total_docs = len(documents)

        if progress_callback:
            progress_callback(0, total_docs, "Initializing vector store...")

        # Clean rebuild
        vector_store = VectorStore(dataset)
        vector_store.reset()

        total_chunks = 0

        for doc_idx, doc in enumerate(documents):
            doc_name = doc["source"]

            if progress_callback:
                progress_callback(
                    doc_idx,
                    total_docs,
                    f"Processing {doc_name}...",
                )

            chunks = chunk_document(
                text=doc["text"],
                dataset=dataset,
                source=doc_name,
                pages=doc.get("pages"),
            )

            if not chunks:
                continue

            # Embed the contextual text (with prefix) for better retrieval
            texts_to_embed = [c["text_to_embed"] for c in chunks]
            embeddings = embed_texts(texts_to_embed)

            # Store raw text (without prefix) in ChromaDB
            vector_store.add_chunks(chunks, embeddings)
            total_chunks += len(chunks)

        if progress_callback:
            progress_callback(total_docs, total_docs, "Done!")

        # Save index metadata
        metadata = IndexMetadata(dataset, config.vectordb_path)
        metadata.save(
            chunk_count=total_chunks,
            ingestion_method="contextual",
        )

        logger.info(
            "Ingestion complete: '%s' → %d chunks", dataset, total_chunks
        )
        return total_chunks

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        dataset: str,
        top_k: Optional[int] = None,
        mode: str = "vector",
    ) -> Tuple[str, List[Dict], str]:
        """Answer a question using RAG.

        Args:
            question: User question.
            dataset: Dataset to search.
            top_k: Number of sources to retrieve.
            mode: Retrieval mode - "vector" (hybrid search), "page_index" (TOC-based),
                  or "pdf_raw" (direct PDF pages)

        Returns:
            (answer_text, sources, full_prompt) — the full_prompt is the
            exact text sent to the LLM for transparency.
        """
        # Choose retriever based on mode
        if mode == "page_index":
            retriever = PageIndexRetriever(dataset)
            sources = retriever.retrieve(question, top_k=top_k or 5)
        elif mode == "pdf_raw":
            retriever = PDFRawRetriever(dataset)
            # Use fewer pages (3) by default to avoid token limits
            sources = retriever.retrieve(question, top_k=top_k or 3)
        else:  # default to vector mode
            retriever = Retriever(dataset, llm=self.llm)
            sources = retriever.retrieve(question, top_k=top_k)

        if not sources:
            return (
                "I couldn't find any relevant information in the selected dataset.",
                [],
                "",
            )

        # Build context with numbered sources
        context_parts = []
        for i, src in enumerate(sources, 1):
            page_info = (
                f" (Page {src['page']})" if src.get("page") else ""
            )
            context_parts.append(
                f"### [Source {i}] {src['source']}{page_info}\n\n{src['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = ANSWER_PROMPT.format(context=context, question=question)
        full_prompt = f"**System:** {SYSTEM_PROMPT}\n\n---\n\n{prompt}"

        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        return answer, sources, full_prompt
