"""Dataset manager – CRUD operations for datasets and their documents."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class DatasetManager:
    """Manage datasets stored on disk (documents + vector indices)."""

    def __init__(
        self,
        documents_root: Optional[str] = None,
        vectordb_root: Optional[str] = None,
    ):
        self.documents_root = documents_root or config.documents_path
        self.vectordb_root = vectordb_root or config.vectordb_path

        os.makedirs(self.documents_root, exist_ok=True)
        os.makedirs(self.vectordb_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset CRUD
    # ------------------------------------------------------------------

    def list_datasets(self) -> List[str]:
        """Return sorted list of dataset names."""
        if not os.path.isdir(self.documents_root):
            return []
        return sorted(
            d
            for d in os.listdir(self.documents_root)
            if os.path.isdir(os.path.join(self.documents_root, d))
        )

    def create_dataset(self, name: str) -> str:
        """Create a new dataset directory. Returns the path."""
        name = self._sanitize_name(name)
        dataset_dir = os.path.join(self.documents_root, name)
        if os.path.exists(dataset_dir):
            raise FileExistsError(f"Dataset '{name}' already exists.")
        os.makedirs(dataset_dir, exist_ok=True)
        logger.info("Created dataset: %s", name)
        return dataset_dir

    def delete_dataset(self, name: str) -> None:
        """Delete a dataset and its associated vector DB."""
        doc_dir = os.path.join(self.documents_root, name)
        vec_dir = os.path.join(self.vectordb_root, name)

        if os.path.isdir(doc_dir):
            shutil.rmtree(doc_dir)
            logger.info("Deleted documents for dataset: %s", name)

        if os.path.isdir(vec_dir):
            shutil.rmtree(vec_dir)
            logger.info("Deleted vector DB for dataset: %s", name)

    def dataset_exists(self, name: str) -> bool:
        return os.path.isdir(os.path.join(self.documents_root, name))

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def list_documents(self, dataset: str) -> List[str]:
        """List document filenames in a dataset."""
        dataset_dir = os.path.join(self.documents_root, dataset)
        if not os.path.isdir(dataset_dir):
            return []
        return sorted(
            f
            for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        )

    def save_uploaded_file(
        self, dataset: str, filename: str, content: bytes
    ) -> str:
        """Save an uploaded file into the dataset directory."""
        max_bytes = config.max_file_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(
                f"File size exceeds limit ({config.max_file_size_mb} MB)."
            )

        dataset_dir = os.path.join(self.documents_root, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("Saved file: %s -> %s", filename, file_path)
        return file_path

    def delete_document(self, dataset: str, filename: str) -> None:
        """Delete a single document from a dataset."""
        file_path = os.path.join(self.documents_root, dataset, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            logger.info("Deleted document: %s/%s", dataset, filename)

    def get_dataset_path(self, dataset: str) -> str:
        return os.path.join(self.documents_root, dataset)

    def get_vectordb_path(self, dataset: str) -> str:
        return os.path.join(self.vectordb_root, dataset)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize dataset name to be filesystem-safe."""
        sanitized = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()
        )
        if not sanitized:
            raise ValueError("Dataset name cannot be empty.")
        return sanitized
