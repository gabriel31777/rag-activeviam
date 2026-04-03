"""
12_index_pdfs_full.py
- Reads raw PDFs using PyMuPDF (fitz) for high-fidelity text + table extraction.
- Auto-detects company name from PDF content (ignores filename).
- Detects header rows like "2021 | 2020 | 2019 | 2018" and stores year metadata.
- Splits into overlapping chunks and indexes into ChromaDB collection: activeviam_pdfs_v1
"""
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import fitz  # pymupdf
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR      = PROJECT_ROOT / "data" / "raw" / "Structured data"
CHROMA_DIR   = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma"

COLLECTION_NAME = "activeviam_pdfs_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 300

# ─── Company keywords → canonical doc name ────────────────────────────────────
COMPANY_KEYWORDS = {
    "absa":       "absa",
    "clicks":     "clicks",
    "distell":    "distell",
    "sasol":      "sasol",
    "pick n pay": "picknpay",
    "picknpay":   "picknpay",
    "pick & pay": "picknpay",
    "tongaat":    "tongaat",
    "impala":     "impala",
    "implats":    "impala",
    "oceana":     "oceana",
    "sibanye":    "ssw",
    "sibanye-stillwater": "ssw",
    "ssw":        "ssw",
    "uct":        "uct",
    "distell":    "distell",
}

YEAR_RE = re.compile(r"\b(20\d{2})\b")


def detect_company(text_sample: str) -> str:
    """Scan the first pages of a PDF for known company keywords."""
    lower = text_sample.lower()
    for kw, canonical in COMPANY_KEYWORDS.items():
        if kw in lower:
            return canonical
    return "unknown"


def detect_report_year(text_sample: str) -> int:
    """Extract the most prominent year mentioned at the start of the PDF."""
    years = [int(y) for y in YEAR_RE.findall(text_sample)]
    if not years:
        return -1
    from collections import Counter
    common = Counter(years).most_common(1)[0][0]
    return common


def extract_page_text(page: fitz.Page) -> str:
    """
    Extract text from a PDF page using 'blocks' mode, which groups
    text by its visual bounding boxes.  This preserves table rows
    much better than raw text mode.
    """
    lines = []
    blocks = page.get_text("blocks")  # each block: (x0,y0,x1,y1, text, block_no, block_type)
    # Sort blocks top-to-bottom left-to-right so reading order is correct
    blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
    for b in blocks_sorted:
        raw = b[4].strip()
        if not raw:
            continue
        # Convert newlines within blocks to pipes (nice table-like format)
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        lines.append(" | ".join(parts) if len(parts) > 1 else parts[0])
    return "\n".join(lines)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks


def index_pdf(pdf_path: Path, collection: chromadb.Collection, doc_name: str, year: int):
    """Extract, chunk and upsert all pages of a PDF."""
    chunks = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc):
            text = extract_page_text(page)
            if not text.strip():
                continue
            # Prepend page reference for context
            page_chunks = chunk_text(f"--- Page {page_num+1} ---\n{text}", CHUNK_SIZE, CHUNK_OVERLAP)
            chunks.extend((page_num, ch) for ch in page_chunks)
        doc.close()
    except Exception as e:
        print(f"  [ERROR] Could not read {pdf_path.name}: {e}")
        return 0

    if not chunks:
        return 0

    batch_ids   = []
    batch_docs  = []
    batch_metas = []

    for idx, (page_num, chunk) in enumerate(chunks):
        # Extract any years found in this specific chunk to help retrieval
        chunk_years = list({int(y) for y in YEAR_RE.findall(chunk)
                            if 2015 <= int(y) <= 2030})
        batch_ids.append(f"{pdf_path.stem}_p{page_num}_c{idx}")
        batch_docs.append(chunk)
        batch_metas.append({
            "doc":    doc_name,
            "year":   year,
            "page":   page_num + 1,
            "source": pdf_path.name,
            "chunk":  idx,
            # Comma-separated string list of years found in chunk
            "years_in_chunk": ",".join(str(y) for y in sorted(chunk_years)),
        })

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(batch_ids), batch_size):
        collection.upsert(
            ids=batch_ids[i:i+batch_size],
            documents=batch_docs[i:i+batch_size],
            metadatas=batch_metas[i:i+batch_size],
        )

    return len(chunks)


def main():
    if not PDF_DIR.exists():
        print(f"[ERROR] PDF directory not found: {PDF_DIR}")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client       = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Always delete old collection and rebuild for cleanness
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted old collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF file(s) in {PDF_DIR}\n")

    total_chunks = 0
    for pdf_path in tqdm(pdf_files, desc="Indexing PDFs"):
        # Read first 3 pages to auto-detect company + year
        try:
            doc = fitz.open(str(pdf_path))
            sample_text = " ".join(
                doc[i].get_text() for i in range(min(4, len(doc)))
            )
            doc.close()
        except Exception:
            sample_text = pdf_path.name

        doc_name = detect_company(sample_text)
        year     = detect_report_year(sample_text[:2000])

        print(f"\n  📄 {pdf_path.name}")
        print(f"     Company: {doc_name}  |  Year: {year}")

        n = index_pdf(pdf_path, collection, doc_name, year)
        print(f"     Chunks indexed: {n}")
        total_chunks += n

    print(f"\n[DONE] Total chunks in '{COLLECTION_NAME}': {collection.count()}")
    print(f"       (Indexed from {len(pdf_files)} PDFs, {total_chunks} chunks total)")


if __name__ == "__main__":
    main()
