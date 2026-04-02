"""Application service for document viewer workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from dataset.loaders import load_document
from dataset.manager import DatasetManager


@dataclass(frozen=True)
class DocumentContent:
    """Structured document content for the viewer page."""

    source: str
    text: str
    pages: Optional[List[Dict]]


class ViewerService:
    """Coordinates document viewer use cases."""

    def __init__(self, manager: Optional[DatasetManager] = None):
        self.manager = manager or DatasetManager()

    def list_datasets(self) -> List[str]:
        """Return available datasets."""
        return self.manager.list_datasets()

    def list_documents(self, dataset: str) -> List[str]:
        """Return documents in a dataset."""
        return self.manager.list_documents(dataset)

    def load_document(self, dataset: str, document: str) -> DocumentContent:
        """Load a document from a dataset."""
        file_path = self.manager.get_document_path(dataset, document)
        doc = load_document(file_path)
        return DocumentContent(
            source=document,
            text=doc["text"],
            pages=doc.get("pages"),
        )
