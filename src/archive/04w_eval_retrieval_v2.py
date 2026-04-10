"""
04w_eval_retrieval_v2.py (version TF-IDF + SVD)
Évaluation avancée du retrieval avec :
  - Match textuel normalisé
  - Match numérique avec tolérance et multiplicateurs d'unités
  - Filtrage par document et année (métadonnées ChromaDB)
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
TOP_K = 5


# =========================
# Expressions régulières
# =========================

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)

# Nombres avec séparateur de milliers par espace : 26 262, 1 234 567
SPACED_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:\s\d{3})+\b")
# Nombres simples : 20835, 95.0, 3.6
SIMPLE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


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
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def normalize_text(s: str) -> str:
    """Normalise un texte pour comparaison (minuscules, sans espaces)."""
    return "".join((s or "").lower().split())


def try_float(x: str) -> Optional[float]:
    """Tente de convertir un texte en float. Retourne None en cas d'échec."""
    try:
        return float(str(x).strip())
    except Exception:
        return None


def extract_numeric_candidates(text: str) -> List[float]:
    """Extrait les candidats numériques d'un texte pour comparaison.

    Inclut :
    - Nombres avec espaces (26 262 -> 26262) + hypothèse décimale (26.262)
    - Nombres simples (20835, 3.6, 95.0, etc.)
    """
    candidates: List[float] = []
    t = text or ""

    # 1) Nombres avec séparateur de milliers par espace
    for m in SPACED_THOUSANDS_RE.finditer(t):
        s = m.group(0)  # ex : "26 262" ou "1 234 567"
        # (a) Comme entier en supprimant les espaces
        as_int = s.replace(" ", "")
        f1 = try_float(as_int)
        if f1 is not None:
            candidates.append(f1)

        # (b) Hypothèse décimale : "26 262" -> "26.262"
        parts = s.split()
        if len(parts) == 2 and len(parts[1]) == 3:
            f2 = try_float(parts[0] + "." + parts[1])
            if f2 is not None:
                candidates.append(f2)

    # 2) Nombres simples
    for m in SIMPLE_NUMBER_RE.finditer(t):
        f = try_float(m.group(0))
        if f is not None:
            candidates.append(f)

    return candidates


def detect_unit_multipliers(text: str) -> List[float]:
    """Détecte les indices d'unité dans le texte et suggère des multiplicateurs.

    Exemples : (Rbn) -> milliards, (Rmn) -> millions.
    """
    t = (text or "").lower()
    multipliers: List[float] = []

    if "rbn" in t or "(rbn" in t:
        multipliers.append(1e9)
    if "bn" in t or "billion" in t:
        multipliers.append(1e9)
    if "mn" in t or "million" in t:
        multipliers.append(1e6)
    if "thousand" in t or "k)" in t or " k " in t:
        multipliers.append(1e3)

    # Supprimer les doublons en gardant l'ordre
    seen = set()
    out = []
    for m in multipliers:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def value_matches(gold_value: str, retrieved_text: str) -> bool:
    """Vérifie si la valeur attendue correspond au texte récupéré.

    Match en deux couches :
    A) Match textuel (normalisé)
    B) Match numérique avec tolérance + multiplicateurs (unités)
    """
    # A) Match textuel
    gv = normalize_text(gold_value)
    rt = normalize_text(retrieved_text)
    if gv and gv in rt:
        return True

    # Tenter aussi la version entière si c'est "1.0" -> "1"
    gf = try_float(gold_value)
    if gf is None:
        return False

    if gf.is_integer():
        if str(int(gf)) in rt:
            return True

    # B) Match numérique robuste
    gold = float(gf)
    candidates = extract_numeric_candidates(retrieved_text)
    if not candidates:
        return False

    # Tolérance : au minimum 1.0, ou 0.5% de la valeur
    tol = max(1.0, abs(gold) * 5e-3)

    # Multiplicateurs : utiliser d'abord ceux détectés, puis les fallbacks
    mults = detect_unit_multipliers(retrieved_text)
    fallback_mults = [1.0, 1e-3, 1e3, 1e6, 1e9]
    all_mults = mults + [m for m in fallback_mults if m not in mults]

    for c in candidates:
        for m in all_mults:
            if abs((c * m) - gold) <= tol:
                return True

    return False


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

    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    collection = build_collection()
    print(f"[INFO] Éléments dans la collection : {collection.count()}")

    hits = 0
    fails_shown = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Évaluation v2"):
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
                print("Filtre utilisé :", where_filter)
                print("Top1 aperçu :", docs[0][:300] if docs else "(vide)")

    rate = hits / len(df)
    print("\n[RESULTATS]")
    print(f"Hit@{TOP_K} (v2) : {rate:.3f} ({hits}/{len(df)})")


if __name__ == "__main__":
    main()
