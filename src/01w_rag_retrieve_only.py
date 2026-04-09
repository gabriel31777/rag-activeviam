"""
01w_rag_retrieve_only.py (version TF-IDF + SVD)
Retrieval seul — charge le CSV, l'indexe dans ChromaDB avec l'embedding
TF-IDF + SVD, puis effectue une requête de test.

Ce script sert de vérification rapide du pipeline de retrieval.
"""

from __future__ import annotations

import os
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
ADD_BATCH_SIZE = 500


# =========================
# Fonctions utilitaires
# =========================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Charge et valide le CSV du dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}\n"
            "Vérifiez que le fichier existe et que le chemin est correct."
        )

    df = pd.read_csv(csv_path)

    # Nettoyage : supprimer les colonnes auto-générées
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    # Vérifier les colonnes requises
    required = {"Question", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Le CSV ne contient pas les colonnes requises : {missing}. "
            f"Colonnes actuelles : {list(df.columns)}"
        )

    # Supprimer les lignes avec un Context vide
    df["Context"] = df["Context"].astype(str)
    df = df[df["Context"].str.strip().ne("")].reset_index(drop=True)

    return df


def make_documents(df: pd.DataFrame) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Prépare les documents, IDs et métadonnées pour l'indexation."""
    documents: List[str] = df["Context"].astype(str).tolist()
    ids: List[str] = [f"ctx_{i:06d}" for i in range(len(documents))]

    questions = df["Question"].astype(str).tolist() if "Question" in df.columns else [""] * len(documents)

    metadatas: List[Dict[str, Any]] = [
        {"row_index": int(i), "question": questions[i]}
        for i in range(len(documents))
    ]

    return documents, ids, metadatas


def build_or_load_collection(embedding_fn: TfidfSvdEmbeddingFunction) -> Any:
    """Crée ou charge la collection ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_to_collection_in_batches(
    collection: Any,
    documents: List[str],
    ids: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Ajoute les documents par batch pour éviter les limites internes de ChromaDB."""
    current_count = collection.count()
    if current_count > 0:
        print(f"[INFO] La collection contient déjà {current_count} éléments. Indexation ignorée.")
        print("       Pour ré-indexer, supprimez le dossier chroma_w2v ou changez COLLECTION_NAME.")
        return

    print(f"[INFO] Indexation de {len(documents)} documents par batch de {ADD_BATCH_SIZE}...")

    for start in tqdm(range(0, len(documents), ADD_BATCH_SIZE), desc="Indexation"):
        end = min(start + ADD_BATCH_SIZE, len(documents))
        collection.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

    print("[INFO] Indexation terminée.")


def retrieve(collection: Any, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Effectue une recherche vectorielle et retourne les résultats formatés."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    ids_ = results["ids"][0]

    out: List[Dict[str, Any]] = []
    for doc, dist, meta, _id in zip(docs, dists, metas, ids_):
        out.append({
            "id": _id,
            "distance": float(dist),
            "metadata": meta,
            "document_preview": (doc[:300] + "..." if len(doc) > 300 else doc),
        })
    return out


# =========================
# Point d'entrée
# =========================

def main():
    print("[INFO] Projet :", PROJECT_ROOT)
    print("[INFO] CSV :", CSV_PATH)
    print("[INFO] Répertoire Chroma :", CHROMA_DIR)
    print("[INFO] Chemin du modèle :", MODEL_PATH)

    # Charger le modèle d'embedding
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle TF-IDF/SVD introuvable : {MODEL_PATH}. "
            "Exécutez d'abord 02c_train_word2vec_pdf.py."
        )

    embedding_fn = TfidfSvdEmbeddingFunction(MODEL_PATH)
    print("[INFO] Modèle TF-IDF/SVD chargé")

    df = load_dataset(CSV_PATH)
    print(f"[INFO] Dataset chargé : {len(df)} lignes | colonnes : {list(df.columns)}")

    documents, ids, metadatas = make_documents(df)
    collection = build_or_load_collection(embedding_fn)
    add_to_collection_in_batches(collection, documents, ids, metadatas)

    print(f"[INFO] Total dans la collection : {collection.count()}")

    # Requête de test
    test_query = "What is the main financial value mentioned?"
    print(f"\n[TEST] Requête : {test_query}")

    hits = retrieve(collection, test_query, top_k=TOP_K)
    for i, h in enumerate(hits, 1):
        print(f"\n--- Résultat {i} ---")
        print("ID :", h["id"])
        print("Distance :", h["distance"])
        print("Métadonnées :", h["metadata"])
        print("Aperçu :", h["document_preview"])


if __name__ == "__main__":
    main()
