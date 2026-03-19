"""Index metadata tracker – tracks ingestion method and compatibility.

This module stores metadata about how vector indices were created,
allowing the UI to warn users when a rebuild is needed for new search modes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Current index version - increment when changing ingestion logic
INDEX_VERSION = "2.0"  # v2.0 supports all search modes (vector, page_index, pdf_raw)

# Metadata file stored alongside each dataset's vector index
METADATA_FILENAME = ".index_metadata.json"


class IndexMetadata:
    """Manages metadata for a dataset's vector index."""

    def __init__(self, dataset: str, vectordb_root: str):
        self.dataset = dataset
        self.metadata_path = os.path.join(vectordb_root, dataset, METADATA_FILENAME)

    def save(
        self,
        version: str = INDEX_VERSION,
        chunk_count: int = 0,
        ingestion_method: str = "standard",
    ) -> None:
        """Save index metadata to disk.

        Args:
            version: Index format version
            chunk_count: Number of chunks indexed
            ingestion_method: Method used for ingestion (e.g., 'standard', 'contextual')
        """
        metadata = {
            "version": version,
            "chunk_count": chunk_count,
            "ingestion_method": ingestion_method,
            "supports_modes": ["vector", "page_index", "pdf_raw"],  # All current modes
        }

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved index metadata for dataset '%s'", self.dataset)

    def load(self) -> Optional[Dict]:
        """Load index metadata from disk.

        Returns:
            Metadata dict or None if not found
        """
        if not os.path.exists(self.metadata_path):
            return None

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load metadata for '%s': %s", self.dataset, e)
            return None

    def is_compatible(self, required_mode: str) -> bool:
        """Check if the index supports a given search mode.

        Args:
            required_mode: Search mode name (e.g., 'vector', 'page_index', 'pdf_raw')

        Returns:
            True if the mode is supported, False otherwise
        """
        metadata = self.load()
        if not metadata:
            # No metadata = old index, only supports vector search
            return required_mode == "vector"

        supported_modes = metadata.get("supports_modes", ["vector"])
        return required_mode in supported_modes

    def needs_rebuild(self, for_modes: list[str]) -> bool:
        """Check if index needs rebuild to support given modes.

        Args:
            for_modes: List of mode names to check

        Returns:
            True if rebuild needed, False if all modes supported
        """
        for mode in for_modes:
            if not self.is_compatible(mode):
                return True
        return False

    def get_status_info(self) -> Dict:
        """Get human-readable status information.

        Returns:
            Dict with status, supported_modes, needs_rebuild flags
        """
        metadata = self.load()

        if not metadata:
            return {
                "status": "⚠️ Old index format",
                "supported_modes": ["vector"],
                "needs_rebuild": True,
                "message": "Rebuild index to use Page Index or PDF Raw modes",
            }

        supported = metadata.get("supports_modes", ["vector"])
        all_modes = ["vector", "page_index", "pdf_raw"]
        missing = [m for m in all_modes if m not in supported]

        if missing:
            return {
                "status": "⚠️ Partial support",
                "supported_modes": supported,
                "needs_rebuild": True,
                "message": f"Rebuild to enable: {', '.join(missing)}",
            }

        return {
            "status": "✅ Full support",
            "supported_modes": supported,
            "needs_rebuild": False,
            "message": "All search modes available",
        }
