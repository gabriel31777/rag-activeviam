"""Contextual document chunking.

Splits documents into retrieval-ready chunks with two key innovations:

1. **Context-aware splitting** – Uses section headers, paragraphs, and
   sentence boundaries to split at natural points.

2. **Contextual prefix** – Each chunk is augmented with a prefix
   containing the source filename, page number, and active section
   hierarchy. This prefix is included when *embedding* the chunk
   (so the embedding captures global context), but stripped when
   presenting the chunk to the LLM for answer generation.

This approach (inspired by Anthropic's "Contextual Retrieval") produces
embeddings that are far more discriminative than raw-chunk embeddings.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Matches a Markdown header line: # Title, ## Subtitle, etc.
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Splitting separators – most natural boundaries first
_SEPARATORS = [
    "\n## ",     # H2 headers
    "\n### ",    # H3 headers
    "\n#### ",   # H4 headers
    "\n\n",      # Paragraphs
    "\n",        # Lines
    ". ",        # Sentences
    " ",         # Words
    "",          # Characters (last resort)
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    dataset: str,
    source: str,
    pages: Optional[List[dict]] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict]:
    """Split a document into retrieval-ready chunks with contextual prefixes.

    Each returned chunk dict has:
        chunk_id       – unique identifier
        text           – the raw chunk content (fed to LLM at query time)
        text_to_embed  – the chunk WITH contextual prefix (used for embedding)
        dataset        – dataset name
        source         – source filename
        page           – page number (int) or None
    """
    _chunk_size = chunk_size or config.chunk_size
    _chunk_overlap = chunk_overlap or config.chunk_overlap

    chunks: List[Dict] = []

    if pages:
        # PDF: chunk each page independently for accurate page citations
        for page_info in pages:
            page_text = page_info["text"]
            page_num = page_info["page"]
            raw_chunks = _split_text(page_text, _chunk_size, _chunk_overlap)

            for piece in raw_chunks:
                piece = piece.strip()
                if not piece:
                    continue
                section_path = _get_section_path(page_text, piece)
                prefix = _build_context_prefix(source, page_num, section_path)

                chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "text": piece,
                        "text_to_embed": f"{prefix}\n\n{piece}",
                        "dataset": dataset,
                        "source": source,
                        "page": page_num,
                    }
                )
    else:
        # Non-PDF: chunk the full text
        raw_chunks = _split_text(text, _chunk_size, _chunk_overlap)

        for piece in raw_chunks:
            piece = piece.strip()
            if not piece:
                continue
            section_path = _get_section_path(text, piece)
            prefix = _build_context_prefix(source, None, section_path)

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": piece,
                    "text_to_embed": f"{prefix}\n\n{piece}",
                    "dataset": dataset,
                    "source": source,
                    "page": None,
                }
            )

    logger.info(
        "Chunked '%s' (%s) -> %d chunks (size=%d, overlap=%d)",
        source,
        dataset,
        len(chunks),
        _chunk_size,
        _chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_text(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Split text using a recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
    )
    return splitter.split_text(text)


def _get_section_path(full_text: str, chunk_text: str) -> str:
    """Determine the section hierarchy for a chunk.

    Locates the chunk within the document and traces back to find
    the active header hierarchy at that position.
    """
    # Use the first 80 chars of the chunk to locate it in the full text
    search_key = chunk_text[:80]
    pos = full_text.find(search_key)
    if pos < 0:
        # Try a shorter prefix
        pos = full_text.find(chunk_text[:40])
    if pos < 0:
        pos = len(full_text)

    text_before = full_text[:pos]
    return _extract_header_hierarchy(text_before)


def _extract_header_hierarchy(text_before: str) -> str:
    """Extract the active header hierarchy at a given point.

    Returns a string like "Annual Report > Financial Data > Revenue".
    When a higher-level header appears, all deeper headers are cleared.
    """
    headers: Dict[int, str] = {}
    for m in _HEADER_RE.finditer(text_before):
        level = len(m.group(1))
        title = m.group(2).strip()
        # Clear deeper headers
        headers = {k: v for k, v in headers.items() if k < level}
        headers[level] = title

    if not headers:
        return ""
    return " > ".join(headers[k] for k in sorted(headers))


def _build_context_prefix(
    source: str,
    page: Optional[int],
    section_path: str,
) -> str:
    """Build a contextual prefix for embedding.

    Example: [Source: report.pdf | Page: 3 | Section: Finance > Revenue]
    """
    parts = [f"[Source: {source}"]
    if page is not None:
        parts.append(f" | Page: {page}")
    if section_path:
        parts.append(f" | Section: {section_path}")
    parts.append("]")
    return "".join(parts)
