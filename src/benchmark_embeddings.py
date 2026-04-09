"""
benchmark_embeddings.py
Compare les performances de retrieval entre TF-IDF + SVD et SentenceTransformers.

Ce script lance les deux evaluations et presente les resultats cote a cote.

Prerequis :
  1. Avoir entraine le modele TF-IDF/SVD : python src/02c_train_word2vec_pdf.py
  2. Avoir indexe avec TF-IDF/SVD :       python src/03w_index_chunks_meta.py
  3. Avoir indexe avec SentenceTransformers : python src/03s_index_chunks_sentence2vec.py

Utilisation :
  python src/benchmark_embeddings.py
"""

from __future__ import annotations

import os
import re
import sys
import time

# Forcer UTF-8 sur Windows pour eviter les erreurs d'encodage
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import chromadb

# Import du module d'embedding partage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import TfidfSvdEmbeddingFunction, SentenceTransformerWrapper

# Import de value_matches
import importlib
eval_v2 = importlib.import_module("04w_eval_retrieval_v2")
value_matches = eval_v2.value_matches
parse_doc_year = eval_v2.parse_doc_year


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# Collections a comparer
CONFIGS = [
    {
        "name": "TF-IDF + SVD",
        "chroma_dir": Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v",
        "collection_name": "data_ret_v3_full_w2v",
        "embedding_type": "tfidf_svd",
        "model_path": Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl",
    },
    {
        "name": "SentenceTransformers (MiniLM-L6-v2)",
        "chroma_dir": Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_st",
        "collection_name": "data_ret_v3_full_st",
        "embedding_type": "sentence_transformer",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
]

TOP_K = 5


# =========================
# Evaluation
# =========================

def evaluate_collection(
    collection: Any,
    df: pd.DataFrame,
    top_k: int,
) -> dict:
    """Evalue le Hit@K d'une collection sur le dataset."""
    hits = 0
    total = len(df)

    for row in tqdm(df.itertuples(index=False), total=total, leave=False):
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
            n_results=top_k,
            include=["documents"],
        )
        if where_filter is not None:
            kwargs["where"] = where_filter

        try:
            res = collection.query(**kwargs)
            docs = res["documents"][0]
            ok = any(value_matches(gold, d) for d in docs)
        except Exception:
            ok = False

        if ok:
            hits += 1

    return {"hits": hits, "total": total, "rate": hits / total if total > 0 else 0}


# =========================
# Point d'entree
# =========================

def main():
    print("=" * 70)
    print("  BENCHMARK : Comparaison des methodes d'embedding pour le RAG")
    print("=" * 70)
    print(f"\n[INFO] CSV : {CSV_PATH}")
    print(f"[INFO] TOP_K : {TOP_K}\n")

    # Charger le dataset
    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        chroma_dir = cfg["chroma_dir"]
        collection_name = cfg["collection_name"]

        print(f"\n{'─' * 50}")
        print(f"  Evaluation : {name}")
        print(f"  Collection : {collection_name}")
        print(f"{'─' * 50}")

        # Instancier l'embedding
        if cfg["embedding_type"] == "tfidf_svd":
            model_path = cfg["model_path"]
            if not model_path.exists():
                print(f"  [ATTENTION] Modele introuvable : {model_path}")
                print(f"  -> Executez d'abord 02c_train_word2vec_pdf.py")
                results.append({"name": name, "hits": 0, "total": len(df), "rate": 0, "status": "MODELE MANQUANT"})
                continue
            embedding_fn = TfidfSvdEmbeddingFunction(model_path)
        else:
            model_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            embedding_fn = SentenceTransformerWrapper(model_name=model_name)

        # Charger la collection
        if not chroma_dir.exists():
            print(f"  [ATTENTION] Repertoire ChromaDB introuvable : {chroma_dir}")
            results.append({"name": name, "hits": 0, "total": len(df), "rate": 0, "status": "NON INDEXE"})
            continue

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        count = collection.count()
        if count == 0:
            print(f"  [ATTENTION] Collection vide. Indexez d'abord les donnees.")
            results.append({"name": name, "hits": 0, "total": len(df), "rate": 0, "status": "COLLECTION VIDE"})
            continue

        print(f"  Elements dans la collection : {count}")

        # Evaluer
        start_time = time.time()
        result = evaluate_collection(collection, df, TOP_K)
        elapsed = time.time() - start_time

        result["name"] = name
        result["status"] = "OK"
        result["time_s"] = elapsed
        results.append(result)

        print(f"  Hit@{TOP_K} : {result['rate']:.3f} ({result['hits']}/{result['total']})")
        print(f"  Temps : {elapsed:.1f}s")

    # Resume comparatif
    print("\n\n" + "=" * 70)
    print("  RESUME COMPARATIF")
    print("=" * 70)
    print(f"\n{'Methode':<45} {'Hit@K':>8} {'Precision':>10} {'Temps':>8} {'Statut':>12}")
    print("─" * 85)

    for r in results:
        name = r["name"]
        hits_str = f"{r['hits']}/{r['total']}" if r["status"] == "OK" else "—"
        rate_str = f"{r['rate']*100:.1f}%" if r["status"] == "OK" else "—"
        time_str = f"{r.get('time_s', 0):.1f}s" if r["status"] == "OK" else "—"
        print(f"  {name:<43} {hits_str:>8} {rate_str:>10} {time_str:>8} {r['status']:>12}")

    print("─" * 85)
    print()


if __name__ == "__main__":
    main()
