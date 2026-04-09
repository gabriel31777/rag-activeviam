"""
04s_eval_retrieval_sentence2vec.py
Evaluation du retrieval avec SentenceTransformers (all-MiniLM-L6-v2).

Meme logique que 04w_eval_retrieval_v2.py (version TF-IDF + SVD),
mais utilise l'embedding SentenceTransformers pour comparaison directe.
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

# Import du module d'embedding partage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import SentenceTransformerWrapper

# Import de la fonction value_matches depuis le module v2
import importlib
eval_v2 = importlib.import_module("04w_eval_retrieval_v2")
value_matches = eval_v2.value_matches
parse_doc_year = eval_v2.parse_doc_year


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_st"
COLLECTION_NAME = "data_ret_v3_full_st"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# =========================
# Fonctions
# =========================

def build_collection() -> Any:
    """Charge la collection ChromaDB avec SentenceTransformers."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerWrapper(model_name=EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# =========================
# Point d'entree
# =========================

def main():
    print("[INFO] CSV :", CSV_PATH)
    print("[INFO] Repertoire Chroma :", CHROMA_DIR)
    print("[INFO] Collection :", COLLECTION_NAME)
    print("[INFO] Modele :", EMBEDDING_MODEL)
    print("[INFO] TOP_K :", TOP_K)

    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    collection = build_collection()
    print(f"[INFO] Elements dans la collection : {collection.count()}")

    if collection.count() == 0:
        print("[ERREUR] La collection est vide. Executez d'abord 03s_index_chunks_sentence2vec.py.")
        return

    hits = 0
    fails_shown = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluation ST"):
        q = str(row.Question)
        gold = str(row.Value)

        doc, year = parse_doc_year(q)

        where_filter = None
        if doc is not None and year is not None:
            where_filter = {
                "$and": [
                    {"doc": {"$eq": doc}},
                    {"year": {"$eq": int(year)}},
                ]
            }

        kwargs = dict(
            query_texts=[q],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"],
        )
        if where_filter is not None:
            kwargs["where"] = where_filter

        res = collection.query(**kwargs)
        docs = res["documents"][0]

        ok = any(value_matches(gold, d) for d in docs)
        if ok:
            hits += 1
        else:
            if fails_shown < 5:
                fails_shown += 1
                print(f"\n--- Echec ---")
                print("Q :", q)
                print("Valeur attendue :", gold)
                print("Filtre utilise :", where_filter)
                print("Top1 apercu :", docs[0][:300] if docs else "(vide)")

    rate = hits / len(df)
    print("\n[RESULTATS — SentenceTransformers]")
    print(f"Hit@{TOP_K} (v2) : {rate:.3f} ({hits}/{len(df)})")


if __name__ == "__main__":
    main()
