"""
02w_eval_retrieval.py (Word2Vec version)
- Avalia a qualidade do retrieval (R do RAG) usando o dataset.
- Para cada linha: query=Question, gold=Value
- Mede Hit@K: se o Value aparece em algum Context recuperado.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

import chromadb


# =========================
# Word2Vec Embedding Function
# =========================

class Word2VecEmbeddingFunction:
    """ChromaDB-compatible embedding function using TF-IDF + SVD."""
    
    def __init__(self, model_path: str | Path):
        """Load pre-trained Word2Vec model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.svd = model_data['svd']
        self.vector_size = model_data.get('vector_size', 300)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not input:
            return []
        
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings_array = self.svd.transform(tfidf_vecs)
        embeddings = [list(row) for row in embeddings_array]
        return embeddings
    
    def embed_query(self, input: str) -> List[float]:
        """Embed a single query."""
        if isinstance(input, list):
            return self(input)[0]
        else:
            return self([input])[0]
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return "word2vec_tfidf_svd"


# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

CHROMA_DIR = Path(os.environ["LOCALAPPDATA"]) / "rag-activiam" / "chroma_w2v"
COLLECTION_NAME = "data_ret_contexts_v1_w2v"
MODEL_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

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
    embedding_fn = Word2VecEmbeddingFunction(MODEL_PATH)

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
    print("[INFO] Model path:", MODEL_PATH)
    print("[INFO] TOP_K:", TOP_K)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Word2Vec model not found at {MODEL_PATH}. Run 02c_train_word2vec_pdf.py first.")

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
