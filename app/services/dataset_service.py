"""Application service for dataset management workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from core.index_metadata import IndexMetadata
from core.rag_pipeline import RAGPipeline
from core.vector_store import VectorStore
from dataset.manager import DatasetManager
from utils.config import get_config


@dataclass(frozen=True)
class DatasetStatus:
    """Dataset state returned to the UI."""

    name: str
    documents: List[str]
    chunk_count: int
    index_message: str
    index_ok: bool
    needs_rebuild: bool


class DatasetService:
    """Coordinates dataset page use cases."""

    def __init__(
        self,
        manager: Optional[DatasetManager] = None,
        pipeline: Optional[RAGPipeline] = None,
    ):
        self.manager = manager or DatasetManager()
        self.pipeline = pipeline
        self.config = get_config()

    def list_datasets(self) -> List[str]:
        """Return available datasets."""
        return self.manager.list_datasets()

    def create_dataset(self, name: str) -> str:
        """Create a dataset."""
        return self.manager.create_dataset(name)

    def delete_dataset(self, name: str) -> None:
        """Delete a dataset and its associated index."""
        self.manager.delete_dataset(name)

    def upload_document(self, dataset: str, filename: str, content: bytes) -> str:
        """Persist an uploaded document."""
        return self.manager.save_uploaded_file(dataset, filename, content)

    def delete_document(self, dataset: str, filename: str) -> None:
        """Remove a document from a dataset."""
        self.manager.delete_document(dataset, filename)

    def get_dataset_status(self, dataset: str) -> DatasetStatus:
        """Return document list and index compatibility status."""
        documents = self.manager.list_documents(dataset)

        try:
            vector_store = VectorStore(dataset)
            chunk_count = vector_store.count()
            metadata = IndexMetadata(dataset, self.config.vectordb_path)
            status_info = metadata.get_status_info()
            return DatasetStatus(
                name=dataset,
                documents=documents,
                chunk_count=chunk_count,
                index_message=status_info["message"],
                index_ok=not status_info["needs_rebuild"],
                needs_rebuild=status_info["needs_rebuild"],
            )
        except Exception:
            return DatasetStatus(
                name=dataset,
                documents=documents,
                chunk_count=0,
                index_message="Vector index: not built yet",
                index_ok=False,
                needs_rebuild=False,
            )

    def build_index(
        self,
        dataset: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        """Build or rebuild the dataset index."""
        pipeline = self.pipeline or RAGPipeline()
        return pipeline.ingest_dataset(
            dataset,
            progress_callback=progress_callback,
        )
