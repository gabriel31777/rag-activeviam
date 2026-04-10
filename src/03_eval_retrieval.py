"""
03_eval_retrieval.py
Évalue la qualité du retrieval (Hit@K) en utilisant le CSV comme ground truth.

Le CSV fournit les paires (Question, Value) = gabarito.
La busca é feita SOMENTE na base de PDFs indexada no ChromaDB.

Utilisation :
    python src/03_eval_retrieval.py --embedding tfidf_svd
    python src/03_eval_retrieval.py --embedding word2vec
    python src/03_eval_retrieval.py --embedding sentence_transformer
    python src/03_eval_retrieval.py --embedding tfidf_svd --top-k 10
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import get_embedding_function


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"
LOCALAPPDATA = os.environ.get("LOCALAPPDATA", ".")
MODELS_DIR = Path(LOCALAPPDATA) / "rag-activeviam" / "models"

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)

# Même configuration que 02_index_pdfs.py
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
# Utilitaires de matching
# =========================

# Nombres avec séparateur de milliers par espace : 26 262, 1 234 567
SPACED_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:\s\d{3})+\b")
# Nombres simples : 20835, 95.0, 3.6
SIMPLE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    """Extrait le nom du document et l'année depuis la question."""
    m = DOC_YEAR_RE.search((question or "").strip())
    if not m:
        return None, None
    return m.group(1).strip().lower(), int(m.group(2))


def normalize_text(s: str) -> str:
    """Normalise un texte pour comparaison."""
    return "".join((s or "").lower().split())


def try_float(x: str) -> Optional[float]:
    """Tente de convertir en float."""
    try:
        return float(str(x).strip())
    except Exception:
        return None


def extract_numeric_candidates(text: str) -> List[float]:
    """Extrait les candidats numériques d'un texte."""
    candidates: List[float] = []
    t = text or ""

    for m in SPACED_THOUSANDS_RE.finditer(t):
        s = m.group(0)
        f1 = try_float(s.replace(" ", ""))
        if f1 is not None:
            candidates.append(f1)
        parts = s.split()
        if len(parts) == 2 and len(parts[1]) == 3:
            f2 = try_float(parts[0] + "." + parts[1])
            if f2 is not None:
                candidates.append(f2)

    for m in SIMPLE_NUMBER_RE.finditer(t):
        f = try_float(m.group(0))
        if f is not None:
            candidates.append(f)

    return candidates


def detect_unit_multipliers(text: str) -> List[float]:
    """Détecte les multiplicateurs d'unités dans le texte."""
    t = (text or "").lower()
    mults = []
    if "rbn" in t or "(rbn" in t or "bn" in t or "billion" in t:
        mults.append(1e9)
    if "mn" in t or "million" in t or "rmn" in t or "(rmn" in t:
        mults.append(1e6)
    if "thousand" in t or "k)" in t or " k " in t:
        mults.append(1e3)
    return list(dict.fromkeys(mults))  # Dédupliqué


def value_matches(gold_value: str, retrieved_text: str) -> bool:
    """Vérifie si la valeur attendue est trouvée dans le texte récupéré."""
    # A) Match textuel
    gv = normalize_text(gold_value)
    rt = normalize_text(retrieved_text)
    if gv and gv in rt:
        return True

    gf = try_float(gold_value)
    if gf is None:
        return False

    if gf.is_integer() and str(int(gf)) in rt:
        return True

    # B) Match numérique
    gold = float(gf)
    candidates = extract_numeric_candidates(retrieved_text)
    if not candidates:
        return False

    tol = max(1.0, abs(gold) * 5e-3)

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
    ap = argparse.ArgumentParser(description="Évaluer le retrieval (Hit@K)")
    ap.add_argument(
        "--embedding",
        choices=["tfidf_svd", "word2vec", "sentence_transformer", "hybrid"],
        default="tfidf_svd",
    )
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    if args.embedding == "hybrid":
        emb_fn_tfidf = get_embedding_function(
            "tfidf_svd", model_path=str(EMBEDDING_CONFIG["tfidf_svd"]["model_path"])
        )
        client_tfidf = chromadb.PersistentClient(path=str(EMBEDDING_CONFIG["tfidf_svd"]["chroma_dir"]))
        col_tfidf = client_tfidf.get_or_create_collection(
            name=EMBEDDING_CONFIG["tfidf_svd"]["collection"],
            embedding_function=emb_fn_tfidf
        )
        
        emb_fn_st = get_embedding_function("sentence_transformer")
        client_st = chromadb.PersistentClient(path=str(EMBEDDING_CONFIG["sentence_transformer"]["chroma_dir"]))
        col_st = client_st.get_or_create_collection(
            name=EMBEDDING_CONFIG["sentence_transformer"]["collection"],
            embedding_function=emb_fn_st
        )
        print(f"[INFO] Mode hybride activé (TF-IDF: {col_tfidf.count()}, ST: {col_st.count()})")
        
        if col_tfidf.count() == 0 or col_st.count() == 0:
            print("[ERREUR] Il manque des index pour le mode hybride !")
            return
    else:
        config = EMBEDDING_CONFIG[args.embedding]
        chroma_dir = config["chroma_dir"]
        collection_name = config["collection"]
        model_path = config["model_path"]

        print(f"[INFO] Embedding   : {args.embedding}")
        print(f"[INFO] CSV (gabarito) : {CSV_PATH}")
        print(f"[INFO] ChromaDB    : {chroma_dir}")
        print(f"[INFO] Collection  : {collection_name}")
        print(f"[INFO] TOP_K       : {args.top_k}")

        if model_path and not Path(model_path).exists():
            print(f"[ERREUR] Modèle introuvable : {model_path}")
            print("         Exécutez d'abord : python src/01_train_embeddings.py")
            return

        emb_fn = get_embedding_function(
            args.embedding,
            model_path=str(model_path) if model_path else None,
        )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[INFO] Éléments dans la collection : {collection.count()}")

        if collection.count() == 0:
            print("[ERREUR] Collection vide ! Exécutez d'abord :")
            print(f"         python src/02_index_pdfs.py --embedding {args.embedding}")
            return

    # Charger le CSV (gabarito)
    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    hits = 0
    fails_shown = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Évaluation"):
        q = str(row.Question)
        gold = str(row.Value)

        doc, year = parse_doc_year(q)

        where_filter = None
        if doc is not None:
            where_filter = {"doc": {"$eq": doc}}

        kwargs = dict(
            query_texts=[q],
            n_results=args.top_k,
            include=["documents", "distances", "metadatas"],
        )
        if where_filter is not None:
            kwargs["where"] = where_filter

        if args.embedding == "hybrid":
            # Reciprocal Rank Fusion (RRF)
            res1 = col_tfidf.query(**kwargs)
            res2 = col_st.query(**kwargs)
            
            scores = {}
            doc_map = {}
            
            for rank, (txt, meta) in enumerate(zip(res1["documents"][0], res1["metadatas"][0])):
                uid = f"{meta.get('doc', '')}_{meta.get('page', '')}"
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                
            for rank, (txt, meta) in enumerate(zip(res2["documents"][0], res2["metadatas"][0])):
                uid = f"{meta.get('doc', '')}_{meta.get('page', '')}"
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                
            sorted_uids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            docs = [doc_map[uid] for uid in sorted_uids[:args.top_k]]
        else:
            res = collection.query(**kwargs)
            docs = res["documents"][0]

        ok = any(value_matches(gold, d) for d in docs)
        if ok:
            hits += 1
        else:
            if fails_shown < 5:
                fails_shown += 1
                print(f"\n--- Échec ---")
                print("Q :", q)
                print("Valeur attendue :", gold)
                print("Top1 :", docs[0][:200] if docs else "(vide)")

    rate = hits / len(df) if len(df) > 0 else 0
    print(f"\n{'='*60}")
    print(f"  RÉSULTATS — {args.embedding}")
    print(f"  Hit@{args.top_k} : {rate:.3f} ({hits}/{len(df)})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
