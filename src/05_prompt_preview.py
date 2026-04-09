"""
05_prompt_preview.py
- Carrega uma pergunta do dataset (por índice ou texto)
- Faz retrieve no Chroma v2 (chunks + metadados)
- Mostra os TOP_K chunks recuperados
- Monta e imprime o prompt final (contexto + pergunta)
"""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import Any, Optional, Tuple, Dict, List

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---- Config igual ao seu v2 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"
CHROMA_DIR = Path(os.environ["LOCALAPPDATA"]) / "rag-activeviam" / "chroma"

COLLECTION_NAME = "data_ret_contexts_v2_chunks_meta"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_DEFAULT = 5

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)


def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    q = (question or "").strip()
    m = DOC_YEAR_RE.search(q)
    if not m:
        return None, None
    doc = m.group(1).strip().lower()
    year = int(m.group(2))
    return doc, year


def build_collection() -> Any:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def make_where_filter(doc: Optional[str], year: Optional[int]) -> Optional[Dict[str, Any]]:
    if doc is None or year is None:
        return None
    # Chroma exige UM operador no topo -> usamos $and
    return {
        "$and": [
            {"doc": {"$eq": doc}},
            {"year": {"$eq": int(year)}},
        ]
    }


def retrieve(collection, query: str, top_k: int, where_filter: Optional[Dict[str, Any]]):
    # Se o filtro não retornar nada, fazemos fallback sem filtro.
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids_ = res["ids"][0]  # ids vêm sempre, não precisa colocar no include

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
        return docs, metas, dists, ids_, None  # filtro efetivo = None (fallback)

    return docs, metas, dists, ids_, where_filter


def build_prompt(question: str, contexts: List[str]) -> str:
    joined = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a careful analyst. Answer ONLY using the information in the contexts.\n"
        "If the answer is not present, say: \"I don't know based on the provided context\".\n\n"
        f"{joined}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=0, help="Índice da linha no CSV (0..)")
    parser.add_argument("--q", type=str, default=None, help="Pergunta exata (se quiser ignorar --i)")
    parser.add_argument("--k", type=int, default=TOP_K_DEFAULT, help="TOP_K")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    if args.q is not None:
        # pega a primeira linha que bate exatamente
        matches = df.index[df["Question"].astype(str) == args.q].tolist()
        if not matches:
            raise ValueError("Não achei essa pergunta exatamente no CSV. Tente usar --i.")
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

    print("\n[INFO] Question:", question)
    print("[INFO] Gold Value:", gold)
    print("[INFO] Parsed doc/year:", (doc, year))
    print("[INFO] Where filter used:", effective_filter)

    print("\n[RETRIEVED]")
    for i, (doc_text, meta, dist, _id) in enumerate(zip(docs, metas, dists, ids_), 1):
        preview = doc_text[:350].replace("\n", " ")
        print(f"\n--- Hit {i} ---")
        print("ID:", _id)
        print("Distance:", float(dist))
        print("Meta:", meta)
        print("Preview:", preview)

    prompt = build_prompt(question, docs[: min(len(docs), args.k)])

    print("\n" + "=" * 80)
    print("[PROMPT FINAL (para enviar ao LLM)]\n")
    print(prompt)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
