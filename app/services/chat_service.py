"""Application service for chat-related operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.rag_pipeline import RAGPipeline
from dataset.manager import DatasetManager


@dataclass(frozen=True)
class SearchModeOption:
    """Presentation metadata for a retrieval mode."""

    key: str
    label: str
    help_text: str
    default_top_k: int


SEARCH_MODE_OPTIONS: Tuple[SearchModeOption, ...] = (
    SearchModeOption(
        key="vector",
        label="🔍 Vector Search (Hybrid + Reranking)",
        help_text="Advanced semantic search with BM25 and reranking.",
        default_top_k=5,
    ),
    SearchModeOption(
        key="page_index",
        label="📑 Page Index (TOC Navigation)",
        help_text="Navigate document structure through extracted headings.",
        default_top_k=5,
    ),
    SearchModeOption(
        key="pdf_raw",
        label="📄 PDF Raw (Direct Page Retrieval)",
        help_text="Retrieve full PDF pages directly to preserve raw text.",
        default_top_k=3,
    ),
)


class ChatService:
    """Coordinates chat page use cases."""

    def __init__(
        self,
        manager: Optional[DatasetManager] = None,
        pipeline: Optional[RAGPipeline] = None,
    ):
        self.manager = manager or DatasetManager()
        self.pipeline = pipeline or RAGPipeline()

    def list_datasets(self) -> List[str]:
        """Return available dataset names."""
        return self.manager.list_datasets()

    def get_search_modes(self) -> Tuple[SearchModeOption, ...]:
        """Return configured retrieval modes for the UI."""
        return SEARCH_MODE_OPTIONS

    def get_search_mode(self, mode: str) -> SearchModeOption:
        """Return metadata for a single retrieval mode."""
        for option in SEARCH_MODE_OPTIONS:
            if option.key == mode:
                return option
        raise ValueError(f"Unsupported search mode: {mode}")

    def ask(
        self,
        question: str,
        dataset: str,
        top_k: Optional[int] = None,
        mode: str = "vector",
    ) -> Dict:
        """Execute a RAG query and normalize the response for the UI."""
        answer, sources, full_prompt = self.pipeline.query(
            question,
            dataset,
            top_k=top_k,
            mode=mode,
        )
        return {
            "answer": answer,
            "sources": sources,
            "prompt": full_prompt,
        }
