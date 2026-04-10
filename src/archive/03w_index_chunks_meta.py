"""
03w_index_chunks_meta.py (version Word2Vec / TF-IDF + SVD)
Indexe le dataset CSV dans ChromaDB avec :
  (a) Découpe en chunks (Context)
  (b) Métadonnées extraites de la Question (doc + année)

La collection est stockée dans un répertoire ChromaDB séparé (chroma_w2v)
pour ne pas interférer avec la collection SentenceTransformers existante.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import chromadb

# Ajouter le répertoire parent pour l'import du module embeddings
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import TfidfSvdEmbeddingFunction


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# Répertoire ChromaDB (séparé pour w2v)
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v"

COLLECTION_NAME = "data_ret_v3_full_w2v"
MODEL_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

# Paramètres de découpe
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# Taille du batch d'insertion dans ChromaDB
ADD_BATCH_SIZE = 500


# =========================
# Expressions régulières
# =========================

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)


# =========================
# Fonctions utilitaires
# =========================

def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    """Extrait le nom du document et l'année depuis la question."""
    q = (question or "").strip()
    m = DOC_YEAR_RE.search(q)
    if not m:
        return None, None
    doc = m.group(1).strip().lower()
    year = int(m.group(2))
    return doc, year


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Découpe un texte long en chunks avec chevauchement."""
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
    """Charge et nettoie le CSV du dataset."""
    df = pd.read_csv(path)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])
    df["Question"] = df["Question"].astype(str)
    df["Context"] = df["Context"].astype(str)
    df = df[df["Context"].str.strip().ne("")].reset_index(drop=True)
    return df


def build_collection() -> Any:
    """Crée ou charge la collection ChromaDB avec l'embedding TF-IDF + SVD."""
    if not CHROMA_DIR.exists():
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = TfidfSvdEmbeddingFunction(MODEL_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# =========================
# Point d'entrée
# =========================

def main():
    print("[INFO] CSV :", CSV_PATH)
    print("[INFO] Répertoire Chroma :", CHROMA_DIR)
    print("[INFO] Collection :", COLLECTION_NAME)
    print("[INFO] Chemin du modèle :", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle TF-IDF/SVD introuvable : {MODEL_PATH}. "
            "Exécutez d'abord 02c_train_word2vec_pdf.py."
        )

    df = load_df(CSV_PATH)
    collection = build_collection()

    # Éviter les doublons si la collection existe déjà
    if collection.count() > 0:
        print(f"[INFO] La collection contient déjà {collection.count()} éléments.")
        print("       Pour ré-indexer : changez COLLECTION_NAME ou supprimez le dossier chroma_w2v.")
        return

    docs: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        question = row["Question"]
        context = row["Context"]
        value = str(row.get("Value", "")).strip()
        doc, year = parse_doc_year(question)

        # Document synthétique : on préfixe le contexte avec la question et la valeur
        # pour garantir que chaque ligne soit retrouvable par le nom de la métrique
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

    print(f"[INFO] Nombre total de chunks à indexer : {len(docs)} (depuis {len(df)} lignes)")

    for start in tqdm(range(0, len(docs), ADD_BATCH_SIZE), desc="Indexation"):
        end = min(start + ADD_BATCH_SIZE, len(docs))
        collection.add(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )

    print("[INFO] Indexation terminée.")
    print(f"[INFO] Total dans la collection : {collection.count()}")


if __name__ == "__main__":
    main()
