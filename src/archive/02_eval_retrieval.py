"""
02_eval_retrieval.py
- Avalia a qualidade do retrieval (R do RAG) usando o dataset.
- Para cada linha: query=Question, gold=Value
- Mede Hit@K: se o Value aparece em algum Context recuperado.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

CHROMA_DIR = Path(os.environ["LOCALAPPDATA"]) / "rag-activeviam" / "chroma"
COLLECTION_NAME = "data_ret_contexts_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
LIMIT = 0  # 0 = usa tudo. Se quiser testar rápido, coloque por ex. 200

# =========================
# Helpers
# =========================

def normalize_text(s: str) -> str:
    """Normalização leve para comparar Value vs Context apesar de formatação diferente."""
    s = str(s).lower().strip()
    s = s.replace("\u00a0", " ")               # non-breaking space
    s = re.sub(r"\s+", " ", s)                 # colapsa espaços
    s = s.replace(",", "")                     # remove separador de milhar (muito comum)
    return s

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    required = {"Question", "Context", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV não tem colunas obrigatórias: {missing}. Colunas atuais: {list(df.columns)}")

    df["Question"] = df["Question"].astype(str)
    df["Context"] = df["Context"].astype(str)
    df["Value"] = df["Value"].astype(str)

    df = df[df["Question"].str.strip().ne("")]
    df = df[df["Context"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    return df

def build_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

def retrieve_docs(collection, query: str, top_k: int) -> List[str]:
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents"],
    )
    return res["documents"][0]

# =========================
# Main
# =========================

def main():
    print("[INFO] CSV:", CSV_PATH)
    print("[INFO] Chroma dir:", CHROMA_DIR)
    print("[INFO] Collection:", COLLECTION_NAME)
    print("[INFO] TOP_K:", TOP_K)

    df = load_dataset(CSV_PATH)
    if LIMIT and LIMIT > 0:
        df = df.head(LIMIT).copy()
        print("[INFO] LIMIT ativo:", LIMIT)

    collection = build_collection()
    print("[INFO] Itens na coleção:", collection.count())

    strict_hits = 0
    norm_hits = 0

    examples_fail: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = row["Question"]
        gold = row["Value"]

        docs = retrieve_docs(collection, q, TOP_K)

        # strict: substring literal
        strict_ok = any(gold.strip() in d for d in docs)

        # normalized: remove diferenças comuns de formatação
        gold_n = normalize_text(gold)
        docs_n = [normalize_text(d) for d in docs]
        norm_ok = any(gold_n in d for d in docs_n)

        strict_hits += int(strict_ok)
        norm_hits += int(norm_ok)

        if (not norm_ok) and len(examples_fail) < 5:
            examples_fail.append({
                "question": q,
                "gold_value": gold,
                "top1_preview": docs[0][:250].replace("\n", " ")
            })

    n = len(df)
    print("\n[RESULTS]")
    print(f"Hit@{TOP_K} (strict)     : {strict_hits/n:.3f}  ({strict_hits}/{n})")
    print(f"Hit@{TOP_K} (normalized) : {norm_hits/n:.3f}  ({norm_hits}/{n})")

    if examples_fail:
        print("\n[EXEMPLOS de falha (até 5)]")
        for i, ex in enumerate(examples_fail, 1):
            print(f"\n--- Fail {i} ---")
            print("Q:", ex["question"])
            print("Gold Value:", ex["gold_value"])
            print("Top1 preview:", ex["top1_preview"])

if __name__ == "__main__":
    main()
