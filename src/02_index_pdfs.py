"""
02_index_pdfs.py
Indexe les PDFs dans ChromaDB en utilisant un embedding au choix.

Source de données : PDFs bruts (data/raw/Structured data/)
Le CSV n'est PAS utilisé ici — il sert uniquement pour l'évaluation.

Utilisation :
    python src/02_index_pdfs.py --embedding tfidf_svd
    python src/02_index_pdfs.py --embedding word2vec
    python src/02_index_pdfs.py --embedding sentence_transformer
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from collections import Counter

import fitz  # pymupdf
import chromadb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import get_embedding_function


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "Structured data"
LOCALAPPDATA = os.environ.get("LOCALAPPDATA", ".")
MODELS_DIR = Path(LOCALAPPDATA) / "rag-activeviam" / "models"

# Chunking par page : on garde chaque page entière.
# Seules les pages très longues (> MAX_PAGE_SIZE) sont découpées.
MAX_PAGE_SIZE = 6000
CHUNK_OVERLAP = 500

YEAR_RE = re.compile(r"\b(20\d{2})\b")

# Mots-clés pour détecter la compagnie dans le PDF
COMPANY_KEYWORDS = {
    "absa":               "absa",
    "clicks":             "clicks",
    "distell":            "distell",
    "sasol":              "sasol",
    "pick n pay":         "picknpay",
    "picknpay":           "picknpay",
    "pick & pay":         "picknpay",
    "tongaat":            "tongaat",
    "impala":             "impala",
    "implats":            "impala",
    "oceana":             "oceana",
    "sibanye":            "ssw",
    "sibanye-stillwater": "ssw",
    "ssw":                "ssw",
}

# Noms des collections et chemins selon l'embedding
EMBEDDING_CONFIG = {
    "tfidf_svd": {
        "chroma_dir": Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_tfidf",
        "collection": "pdfs_tfidf_svd",
        "model_path": MODELS_DIR / "tfidf_svd_model.pkl",
    },
    "word2vec": {
        "chroma_dir": Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_w2v",
        "collection": "pdfs_word2vec",
        "model_path": MODELS_DIR / "word2vec_model.pkl",
    },
    "sentence_transformer": {
        "chroma_dir": Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_st",
        "collection": "pdfs_sentence_transformer",
        "model_path": None,
    },
}


# =========================
# Utilitaires
# =========================

def detect_company(text_sample: str) -> str:
    """Détecte le nom de la compagnie à partir du texte."""
    lower = text_sample.lower()
    for kw, canonical in COMPANY_KEYWORDS.items():
        if kw in lower:
            return canonical
    return "unknown"


def detect_report_year(text_sample: str) -> int:
    """Extrait l'année la plus fréquente du début du PDF."""
    years = [int(y) for y in YEAR_RE.findall(text_sample)]
    if not years:
        return -1
    return Counter(years).most_common(1)[0][0]


def extract_page_text(page: fitz.Page) -> str:
    """Extrait le texte d'une page PDF en préservant les tableaux."""
    lines = []
    blocks = page.get_text("blocks")
    blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
    for b in blocks_sorted:
        raw = b[4].strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        lines.append(" | ".join(parts) if len(parts) > 1 else parts[0])
    return "\n".join(lines)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Découpe un texte en chunks avec chevauchement."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


# =========================
# Indexation
# =========================

def index_pdf(
    pdf_path: Path,
    collection: chromadb.Collection,
    doc_name: str,
    year: int,
) -> int:
    """Extrait et insère les pages d'un PDF (chunking par page)."""
    chunks = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc):
            text = extract_page_text(page)
            if not text.strip():
                continue
            page_text = f"--- Page {page_num + 1} ---\n{text}"
            # Garder la page entière si possible, sinon découper
            if len(page_text) <= MAX_PAGE_SIZE:
                chunks.append((page_num, page_text))
            else:
                page_chunks = chunk_text(page_text, MAX_PAGE_SIZE, CHUNK_OVERLAP)
                chunks.extend((page_num, ch) for ch in page_chunks)
        doc.close()
    except Exception as e:
        print(f"  [ERREUR] Impossible de lire {pdf_path.name}: {e}")
        return 0

    if not chunks:
        return 0

    batch_ids = []
    batch_docs = []
    batch_metas = []

    for idx, (page_num, chunk) in enumerate(chunks):
        chunk_years = list(
            {int(y) for y in YEAR_RE.findall(chunk) if 2015 <= int(y) <= 2030}
        )
        batch_ids.append(f"{pdf_path.stem}_p{page_num}_c{idx}")
        batch_docs.append(chunk)
        batch_metas.append({
            "doc": doc_name,
            "year": year,
            "page": page_num + 1,
            "source": pdf_path.name,
            "chunk": idx,
            "years_in_chunk": ",".join(str(y) for y in sorted(chunk_years)),
        })

    # Insertion par batch de 100
    for i in range(0, len(batch_ids), 100):
        collection.upsert(
            ids=batch_ids[i:i + 100],
            documents=batch_docs[i:i + 100],
            metadatas=batch_metas[i:i + 100],
        )

    return len(chunks)


# =========================
# Point d'entrée
# =========================

def main():
    ap = argparse.ArgumentParser(description="Indexer les PDFs dans ChromaDB")
    ap.add_argument(
        "--embedding",
        choices=["tfidf_svd", "word2vec", "sentence_transformer"],
        default="tfidf_svd",
        help="Type d'embedding à utiliser",
    )
    ap.add_argument("--force", action="store_true", help="Supprimer et recréer la collection")
    args = ap.parse_args()

    if not PDF_DIR.exists():
        print(f"[ERREUR] Répertoire PDF introuvable : {PDF_DIR}")
        return

    config = EMBEDDING_CONFIG[args.embedding]
    chroma_dir = config["chroma_dir"]
    collection_name = config["collection"]
    model_path = config["model_path"]

    print(f"[INFO] Embedding : {args.embedding}")
    print(f"[INFO] ChromaDB  : {chroma_dir}")
    print(f"[INFO] Collection: {collection_name}")

    # Vérifier que le modèle existe (si nécessaire)
    if model_path and not model_path.exists():
        print(f"[ERREUR] Modèle introuvable : {model_path}")
        print("         Exécutez d'abord : python src/01_train_embeddings.py")
        return

    # Créer l'embedding
    emb_fn = get_embedding_function(
        args.embedding,
        model_path=str(model_path) if model_path else None,
    )

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Supprimer l'ancienne collection si --force
    if args.force:
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] Ancienne collection '{collection_name}' supprimée.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not args.force:
        print(f"[INFO] La collection contient déjà {collection.count()} éléments.")
        print("       Utilisez --force pour supprimer et recréer.")
        return

    # Indexer tous les PDFs
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] {len(pdf_files)} fichiers PDF trouvés\n")

    total_chunks = 0
    for pdf_path in tqdm(pdf_files, desc="Indexation"):
        try:
            doc = fitz.open(str(pdf_path))
            sample_text = " ".join(
                doc[i].get_text() for i in range(min(4, len(doc)))
            )
            doc.close()
        except Exception:
            sample_text = pdf_path.name

        doc_name = detect_company(sample_text)
        year = detect_report_year(sample_text[:2000])

        print(f"\n  📄 {pdf_path.name}")
        print(f"     Compagnie : {doc_name}  |  Année : {year}")

        n = index_pdf(pdf_path, collection, doc_name, year)
        print(f"     Chunks indexés : {n}")
        total_chunks += n

    print(f"\n[OK] Total : {collection.count()} chunks dans '{collection_name}'")
    print(f"     ({len(pdf_files)} PDFs, {total_chunks} chunks)")


if __name__ == "__main__":
    main()
