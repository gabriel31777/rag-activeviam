"""
02w_eval_retrieval.py (version TF-IDF + SVD)
Évalue la qualité du retrieval (le R de RAG) sur le dataset.

Pour chaque ligne du CSV :
  - query  = Question
  - gold   = Value (valeur attendue)
  - Mesure Hit@K : la valeur gold apparaît-elle dans un des contextes récupérés ?
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
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
COLLECTION_NAME = "data_ret_contexts_v1_w2v"
MODEL_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

TOP_K = 5
LIMIT = 0  # 0 = utilise tout le dataset ; sinon limite pour un test rapide


# =========================
# Fonctions utilitaires
# =========================

def normalize_text(s: str) -> str:
    """Normalisation légère pour comparer Value vs Context malgré le formatage."""
    s = str(s).lower().strip()
    s = s.replace("\u00a0", " ")       # espace insécable
    s = re.sub(r"\s+", " ", s)         # collapser les espaces multiples
    s = s.replace(",", "")             # séparateur de milliers
    return s


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Charge et valide le CSV du dataset."""
    df = pd.read_csv(csv_path)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    required = {"Question", "Context", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Le CSV ne contient pas les colonnes requises : {missing}. "
            f"Colonnes actuelles : {list(df.columns)}"
        )

    df["Question"] = df["Question"].astype(str)
    df["Context"] = df["Context"].astype(str)
    df["Value"] = df["Value"].astype(str)

    df = df[df["Question"].str.strip().ne("")]
    df = df[df["Context"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    return df


def build_collection() -> Any:
    """Charge la collection ChromaDB avec l'embedding TF-IDF + SVD."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = TfidfSvdEmbeddingFunction(MODEL_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def retrieve_docs(collection: Any, query: str, top_k: int) -> List[str]:
    """Récupère les documents les plus proches de la requête."""
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"],
    )
    return res["documents"][0]


# =========================
# Point d'entrée
# =========================

def main():
    print("[INFO] CSV :", CSV_PATH)
    print("[INFO] Répertoire Chroma :", CHROMA_DIR)
    print("[INFO] Collection :", COLLECTION_NAME)
    print("[INFO] Chemin du modèle :", MODEL_PATH)
    print("[INFO] TOP_K :", TOP_K)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle TF-IDF/SVD introuvable : {MODEL_PATH}. "
            "Exécutez d'abord 02c_train_word2vec_pdf.py."
        )

    df = load_dataset(CSV_PATH)
    if LIMIT and LIMIT > 0:
        df = df.head(LIMIT).copy()
        print(f"[INFO] LIMIT actif : {LIMIT}")

    collection = build_collection()
    print(f"[INFO] Éléments dans la collection : {collection.count()}")

    strict_hits = 0
    norm_hits = 0
    examples_fail: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Évaluation"):
        q = row["Question"]
        gold = row["Value"]

        docs = retrieve_docs(collection, q, TOP_K)

        # Match strict : sous-chaîne littérale
        strict_ok = any(gold.strip() in d for d in docs)

        # Match normalisé : supprime les différences de formatage courantes
        gold_n = normalize_text(gold)
        docs_n = [normalize_text(d) for d in docs]
        norm_ok = any(gold_n in d for d in docs_n)

        strict_hits += int(strict_ok)
        norm_hits += int(norm_ok)

        if (not norm_ok) and len(examples_fail) < 5:
            examples_fail.append({
                "question": q,
                "gold_value": gold,
                "top1_preview": docs[0][:250].replace("\n", " "),
            })

    n = len(df)
    print("\n[RESULTATS]")
    print(f"Hit@{TOP_K} (strict)     : {strict_hits/n:.3f}  ({strict_hits}/{n})")
    print(f"Hit@{TOP_K} (normalisé)  : {norm_hits/n:.3f}  ({norm_hits}/{n})")

    if examples_fail:
        print(f"\n[EXEMPLES D'ECHEC (max 5)]")
        for i, ex in enumerate(examples_fail, 1):
            print(f"\n--- Echec {i} ---")
            print("Q :", ex["question"])
            print("Valeur attendue :", ex["gold_value"])
            print("Top1 aperçu :", ex["top1_preview"])


if __name__ == "__main__":
    main()
