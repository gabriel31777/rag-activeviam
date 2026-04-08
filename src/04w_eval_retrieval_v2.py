"""
04w_eval_retrieval_v2.py (Word2Vec version)
- Avalia Hit@K na coleção v2 (chunks + metadados)
- Usa filtro por doc/year extraído da Question, quando existir
- Matching de Value mais robusto (texto + números com formatos diferentes)
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

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
CSV_PATH = PROJECT_ROOT / "data" / "Data_ret.csv"
CHROMA_DIR = Path(os.environ["LOCALAPPDATA"]) / "rag-activiam" / "chroma_w2v"

COLLECTION_NAME = "data_ret_contexts_v2_chunks_meta_w2v"
MODEL_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"
TOP_K = 5

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)

# captura números do tipo: 26 262  | 27 873 | 1 234 567
SPACED_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:\s\d{3})+\b")
# captura números simples: 20835, 95.0, 3.6 etc.
SIMPLE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


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
    embedding_fn = Word2VecEmbeddingFunction(MODEL_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def normalize_text(s: str) -> str:
    return "".join((s or "").lower().split())


def try_float(x: str) -> Optional[float]:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def extract_numeric_candidates(text: str) -> List[float]:
    """
    Extrai candidatos numéricos do texto para comparação.
    Inclui:
    - números com espaços (26 262 -> 26262) e também uma hipótese "decimal" (26.262)
    - números simples (20835, 3.6, 95.0, etc.)
    """
    candidates: List[float] = []

    t = text or ""

    # 1) números com separador de milhar por espaço
    for m in SPACED_THOUSANDS_RE.finditer(t):
        s = m.group(0)  # ex "26 262" ou "1 234 567"
        # (a) como inteiro removendo espaços
        as_int = s.replace(" ", "")
        f1 = try_float(as_int)
        if f1 is not None:
            candidates.append(f1)

        # (b) hipótese comum em tabelas: "26 262" querendo dizer "26.262"
        # Só aplica se for exatamente 2 grupos: d{1,3} d{3}
        parts = s.split()
        if len(parts) == 2 and len(parts[1]) == 3:
            f2 = try_float(parts[0] + "." + parts[1])
            if f2 is not None:
                candidates.append(f2)

    # 2) números simples
    for m in SIMPLE_NUMBER_RE.finditer(t):
        f = try_float(m.group(0))
        if f is not None:
            candidates.append(f)

    return candidates


def detect_unit_multipliers(text: str) -> List[float]:
    """
    Detecta pistas de unidade no texto e sugere multiplicadores.
    Ex.: (Rbn) -> bilhões.
    """
    t = (text or "").lower()

    multipliers: List[float] = []

    # pistas fortes
    if "rbn" in t or "(rbn" in t:
        multipliers.append(1e9)
    if "bn" in t or "billion" in t:
        multipliers.append(1e9)
    if "mn" in t or "million" in t:
        multipliers.append(1e6)
    if "thousand" in t or "k)" in t or " k " in t:
        multipliers.append(1e3)

    # remove duplicados mantendo ordem
    seen = set()
    out = []
    for m in multipliers:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def value_matches(gold_value: str, retrieved_text: str) -> bool:
    """
    Match em duas camadas:
    A) Match textual (normalizado)
    B) Match numérico com tolerância + multiplicadores (unidades)
    """
    # A) textual
    gv = normalize_text(gold_value)
    rt = normalize_text(retrieved_text)
    if gv and gv in rt:
        return True

    # tenta também versão inteira se for "1.0" -> "1"
    gf = try_float(gold_value)
    if gf is None:
        return False

    if gf.is_integer():
        if str(int(gf)) in rt:
            return True

    # B) numérico robusto
    gold = float(gf)

    candidates = extract_numeric_candidates(retrieved_text)
    if not candidates:
        return False

    # tolerância: pelo menos 1.0, ou 0.5% do valor (para rounding differences)
    tol = max(1.0, abs(gold) * 5e-3)

    # multiplicadores: se o texto sugere algo, usa primeiro; senão tenta alguns padrões
    mults = detect_unit_multipliers(retrieved_text)
    fallback_mults = [1.0, 1e-3, 1e3, 1e6, 1e9]

    # junta (prioriza os detectados)
    all_mults = mults + [m for m in fallback_mults if m not in mults]

    for c in candidates:
        for m in all_mults:
            if abs((c * m) - gold) <= tol:
                return True

    return False


def main():
    print("[INFO] CSV:", CSV_PATH)
    print("[INFO] Chroma dir:", CHROMA_DIR)
    print("[INFO] Collection:", COLLECTION_NAME)
    print("[INFO] Model path:", MODEL_PATH)
    print("[INFO] TOP_K:", TOP_K)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Word2Vec model not found at {MODEL_PATH}. Run 02c_train_word2vec_pdf.py first.")

    df = pd.read_csv(CSV_PATH)
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    collection = build_collection()
    print("[INFO] Itens na coleção:", collection.count())

    hits = 0
    fails_shown = 0

    for row in tqdm(df.itertuples(index=False), total=len(df)):
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

        # só passa `where=` se existir filtro
        if where_filter is not None:
            res = collection.query(
                query_texts=[q],
                n_results=TOP_K,
                where=where_filter,
                include=["documents", "distances", "metadatas"],
            )
        else:
            res = collection.query(
                query_texts=[q],
                n_results=TOP_K,
                include=["documents", "distances", "metadatas"],
            )

        docs = res["documents"][0]

        ok = any(value_matches(gold, d) for d in docs)
        if ok:
            hits += 1
        else:
            if fails_shown < 5:
                fails_shown += 1
                print("\n--- Fail ---")
                print("Q:", q)
                print("Gold Value:", gold)
                print("Where filter used:", where_filter)
                print("Top1 preview:", docs[0][:300])

    rate = hits / len(df)
    print("\n[RESULTS]")
    print(f"Hit@{TOP_K} (v2) : {rate:.3f} ({hits}/{len(df)})")


if __name__ == "__main__":
    main()
