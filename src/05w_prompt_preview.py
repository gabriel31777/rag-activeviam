"""
05w_prompt_preview.py (version TF-IDF + SVD)
Prévisualisation du prompt envoyé au LLM.

Charge une question du dataset (par index ou texte), effectue le retrieval
dans ChromaDB, et affiche le prompt final complet tel qu'il serait envoyé
au modèle de génération.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import chromadb

# Import du module d'embedding partagé
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import TfidfSvdEmbeddingFunction


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v"

COLLECTION_NAME = "data_ret_contexts_v2_chunks_meta_w2v"
MODEL_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

TOP_K_DEFAULT = 5

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


def build_collection() -> Any:
    """Charge la collection ChromaDB avec l'embedding TF-IDF + SVD."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = TfidfSvdEmbeddingFunction(MODEL_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def make_where_filter(doc: Optional[str], year: Optional[int]) -> Optional[Dict[str, Any]]:
    """Construit le filtre ChromaDB à partir du document et de l'année."""
    if doc is None or year is None:
        return None
    return {
        "$and": [
            {"doc": {"$eq": doc}},
            {"year": {"$eq": int(year)}},
        ]
    }


def retrieve(
    collection: Any,
    query: str,
    top_k: int,
    where_filter: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[Dict], List[float], List[str], Optional[Dict]]:
    """Effectue le retrieval avec fallback si le filtre ne retourne rien."""
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids_ = res["ids"][0]

    # Fallback sans filtre si aucun résultat
    if len(docs) == 0 and where_filter is not None:
        res = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        ids_ = res["ids"][0]
        return docs, metas, dists, ids_, None  # filtre effectif = None (fallback)

    return docs, metas, dists, ids_, where_filter


def build_prompt(question: str, contexts: List[str]) -> str:
    """Construit le prompt final à envoyer au LLM."""
    joined = "\n\n".join([f"[Contexte {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "Vous etes un analyste rigoureux. Repondez UNIQUEMENT en utilisant les informations des contextes.\n"
        "Si la reponse n'est pas presente, dites : \"Je ne sais pas d'apres le contexte fourni\".\n\n"
        f"{joined}\n\n"
        f"Question : {question}\n"
        "Reponse :"
    )


# =========================
# Point d'entree
# =========================

def main():
    parser = argparse.ArgumentParser(description="Previsualisation du prompt RAG")
    parser.add_argument("--i", type=int, default=0, help="Index de la ligne dans le CSV (0..)")
    parser.add_argument("--q", type=str, default=None, help="Question exacte (ignore --i)")
    parser.add_argument("--k", type=int, default=TOP_K_DEFAULT, help="Nombre de resultats (TOP_K)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele TF-IDF/SVD introuvable : {MODEL_PATH}. "
            "Executez d'abord 02c_train_word2vec_pdf.py."
        )

    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    if args.q is not None:
        matches = df.index[df["Question"].astype(str) == args.q].tolist()
        if not matches:
            raise ValueError("Question non trouvee dans le CSV. Essayez --i a la place.")
        idx = matches[0]
    else:
        idx = args.i

    row = df.iloc[idx]
    question = str(row["Question"])
    gold = str(row["Value"])

    doc, year = parse_doc_year(question)
    where_filter = make_where_filter(doc, year)

    collection = build_collection()
    docs, metas, dists, ids_, effective_filter = retrieve(collection, question, args.k, where_filter)

    print(f"\n[INFO] Question : {question}")
    print(f"[INFO] Valeur attendue : {gold}")
    print(f"[INFO] Document/Annee extraits : ({doc}, {year})")
    print(f"[INFO] Filtre utilise : {effective_filter}")

    print("\n[RESULTATS RETRIEVAL]")
    for i, (doc_text, meta, dist, _id) in enumerate(zip(docs, metas, dists, ids_), 1):
        preview = doc_text[:350].replace("\n", " ")
        print(f"\n--- Resultat {i} ---")
        print(f"ID : {_id}")
        print(f"Distance : {float(dist)}")
        print(f"Metadonnees : {meta}")
        print(f"Apercu : {preview}")

    prompt = build_prompt(question, docs[:min(len(docs), args.k)])

    print("\n" + "=" * 80)
    print("[PROMPT FINAL (a envoyer au LLM)]\n")
    print(prompt)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
