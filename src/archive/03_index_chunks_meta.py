"""
03_index_chunks_meta.py
- Indexes the dataset with:
  (a) Context chunking
  (b) Metadata extracted from the Question (doc + year)
- Saves into a new collection within the same CHROMA_DIR
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# Keep path valid on Windows (outside OneDrive to avoid lock issues)
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma"

COLLECTION_NAME = "data_ret_v3_full"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# Insertion settings
ADD_BATCH_SIZE = 500


DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)


def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    q = (question or "").strip()
    m = DOC_YEAR_RE.search(q)
    if not m:
        return None, None
    doc = m.group(1).strip().lower()
    year = int(m.group(2))
    return doc, year


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= size:
        return [t]

    chunks: List[str] = []
    start = 0
    while start < len(t):
        end = min(start + size, len(t))
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(t):
            break
        start = max(0, end - overlap)
    return chunks


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])
    df["Question"] = df["Question"].astype(str)
    df["Context"] = df["Context"].astype(str)
    df = df[df["Context"].str.strip().ne("")].reset_index(drop=True)
    return df


def build_collection() -> Any:
    # Use str(CHROMA_DIR) to ensure compatibility
    if not CHROMA_DIR.exists():
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def main():
    print("[INFO] CSV:", CSV_PATH)
    print("[INFO] Chroma dir:", CHROMA_DIR)
    print("[INFO] Collection:", COLLECTION_NAME)

    df = load_df(CSV_PATH)
    collection = build_collection()

    # If already exists, do not duplicate
    if collection.count() > 0:
        print(f"[INFO] Collection already has {collection.count()} items. Stopping to avoid duplicates.")
        print("       To re-index: change COLLECTION_NAME or delete the collection/chroma folder.")
        return

    docs: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        question = row["Question"]
        context = row["Context"]
        value = str(row.get("Value", "")).strip()
        doc, year = parse_doc_year(question)

        # --- Synthetic anchor doc: guarantees this row is findable by full question ---
        # We prepend the question to the context so even misaligned contexts
        # can be retrieved by the metric name that appears in the question.
        full_text = f"QUESTION: {question}\nVALUE: {value}\n\nCONTEXT:\n{context}"

        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for j, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"row_{i:06d}_chunk_{j:03d}")
            metas.append(
                {
                    "row_index": int(i),
                    "chunk_index": int(j),
                    "doc": doc if doc is not None else "",
                    "year": int(year) if year is not None else -1,
                    "question": question,
                    "value": value,
                }
            )

    print(f"[INFO] Total chunks to index: {len(docs)} (from {len(df)} rows)")

    for start in tqdm(range(0, len(docs), ADD_BATCH_SIZE)):
        end = min(start + ADD_BATCH_SIZE, len(docs))
        collection.add(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )

    print("[INFO] Indexing finished.")
    print("[INFO] Total in collection:", collection.count())


if __name__ == "__main__":
    main()