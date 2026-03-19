"""Page Index RAG – retrieval based on hierarchical document structure.

This approach builds a table of contents (TOC) tree from documents and
navigates it to find relevant sections, as an alternative to pure vector search.

Process:
  1. Build TOC: Extract headings and structure from documents
  2. Index nodes: Create embeddings for each TOC node (heading + summary)
  3. Navigate: Use semantic similarity to traverse the tree
  4. Retrieve: Return content from the most relevant sections
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.embeddings import embed_query, embed_texts
from core.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# TOC Node
# ---------------------------------------------------------------------------


class TOCNode:
    """Node in the table of contents tree."""

    def __init__(
        self,
        level: int,
        title: str,
        content: str = "",
        source: str = "",
        page: Optional[int] = None,
    ):
        self.level = level  # Heading level (1 = h1, 2 = h2, etc.)
        self.title = title
        self.content = content  # Text under this heading
        self.source = source
        self.page = page
        self.children: List[TOCNode] = []
        self.embedding: Optional[np.ndarray] = None

    def add_child(self, child: TOCNode):
        """Add a child node."""
        self.children.append(child)

    def get_text_for_embedding(self) -> str:
        """Get text representation for embedding."""
        # Combine title and a snippet of content
        content_snippet = self.content[:500] if self.content else ""
        return f"{self.title}\n\n{content_snippet}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for retrieval results."""
        return {
            "source": self.source,
            "page": self.page,
            "text": f"# {self.title}\n\n{self.content}",
            "section": self.title,
        }


# ---------------------------------------------------------------------------
# TOC Builder
# ---------------------------------------------------------------------------


class TOCBuilder:
    """Builds a table of contents tree from document text."""

    # Regex to detect markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    @staticmethod
    def build_toc(
        text: str, source: str = "", page: Optional[int] = None
    ) -> List[TOCNode]:
        """Build TOC tree from text with markdown headings.

        Args:
            text: Document text (should contain markdown headings)
            source: Source document name
            page: Page number (if applicable)

        Returns:
            List of root-level TOC nodes
        """
        # Find all headings
        headings = list(TOCBuilder.HEADING_PATTERN.finditer(text))

        if not headings:
            # No structure found, create a single root node
            root = TOCNode(
                level=1,
                title=f"Document: {source}",
                content=text[:2000],  # First 2000 chars
                source=source,
                page=page,
            )
            return [root]

        roots: List[TOCNode] = []
        stack: List[TOCNode] = []

        for i, match in enumerate(headings):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            start_pos = match.end()

            # Extract content until next heading
            if i + 1 < len(headings):
                end_pos = headings[i + 1].start()
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()

            node = TOCNode(
                level=level,
                title=title,
                content=content,
                source=source,
                page=page,
            )

            # Build tree structure
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].add_child(node)
            else:
                roots.append(node)

            stack.append(node)

        return roots

    @staticmethod
    def extract_all_nodes(roots: List[TOCNode]) -> List[TOCNode]:
        """Flatten tree into a list of all nodes."""
        nodes = []

        def traverse(node: TOCNode):
            nodes.append(node)
            for child in node.children:
                traverse(child)

        for root in roots:
            traverse(root)

        return nodes


# ---------------------------------------------------------------------------
# Page Index Retriever
# ---------------------------------------------------------------------------


class PageIndexRetriever:
    """Retriever using hierarchical TOC navigation."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.vector_store = VectorStore(dataset)
        self.toc_nodes: List[TOCNode] = []
        self.toc_embeddings: Optional[np.ndarray] = None
        self._build_index()

    def _build_index(self):
        """Build the TOC index from all documents in the dataset."""
        logger.info("Building page index for dataset '%s'", self.dataset)

        # Get all chunks from vector store
        all_chunks = self.vector_store.get_all_chunks()

        if not all_chunks:
            logger.warning("No chunks found in dataset '%s'", self.dataset)
            return

        # Group chunks by source document
        docs_by_source: Dict[str, List[Dict]] = {}
        for chunk in all_chunks:
            source = chunk.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(chunk)

        # Build TOC for each document
        all_nodes = []
        for source, chunks in docs_by_source.items():
            # Reconstruct document text from chunks
            # Sort by page if available
            chunks_sorted = sorted(
                chunks, key=lambda x: x.get("page", 0) or 0
            )
            full_text = "\n\n".join(c["text"] for c in chunks_sorted)

            # Build TOC tree
            roots = TOCBuilder.build_toc(
                full_text,
                source=source,
                page=chunks_sorted[0].get("page") if chunks_sorted else None,
            )

            # Extract all nodes from tree
            nodes = TOCBuilder.extract_all_nodes(roots)
            all_nodes.extend(nodes)

        self.toc_nodes = all_nodes

        if not self.toc_nodes:
            logger.warning("No TOC nodes created for dataset '%s'", self.dataset)
            return

        # Create embeddings for all TOC nodes
        texts_to_embed = [node.get_text_for_embedding() for node in self.toc_nodes]
        embeddings = embed_texts(texts_to_embed)
        self.toc_embeddings = np.array(embeddings)

        # Store embeddings in nodes
        for node, emb in zip(self.toc_nodes, embeddings):
            node.embedding = np.array(emb)

        logger.info(
            "Page index built: %d TOC nodes from %d documents",
            len(self.toc_nodes),
            len(docs_by_source),
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant sections using TOC navigation.

        Args:
            query: User query
            top_k: Number of sections to return

        Returns:
            List of relevant section dictionaries
        """
        if not self.toc_nodes or self.toc_embeddings is None:
            logger.warning("TOC index not built, returning empty results")
            return []

        # Embed query
        query_embedding = np.array(embed_query(query))

        # Compute similarities to all TOC nodes
        similarities = np.dot(self.toc_embeddings, query_embedding)

        # Get top-k most similar nodes
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            node = self.toc_nodes[idx]
            score = float(similarities[idx])

            result = node.to_dict()
            result["similarity_score"] = score
            results.append(result)

        logger.info(
            "Page index retrieval: %d results for query '%s...'",
            len(results),
            query[:50],
        )

        return results
