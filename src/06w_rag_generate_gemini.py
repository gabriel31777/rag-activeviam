"""
06w_rag_generate_gemini.py (Word2Vec version)
- RAG (retrieve + generate) using ChromaDB (v2 chunks + metadata) + Gemini
- Final Robust Version:
  1. Reconstructs fragmented tables by crossing different chunks.
  2. Injects Year and Document metadata into the prompt.
  3. Uses a wide context window (k=15) and maximum output (8k tokens).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

import chromadb

# Google Gemini (new SDK)
from google import genai
from google.genai import types


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
# Paths / Config defaults
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH_DEFAULT = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# On Windows, avoid OneDrive for persistent index
LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR_DEFAULT = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_w2v" if LOCALAPPDATA else (PROJECT_ROOT / "data" / "chroma_w2v")
)

COLLECTION_DEFAULT = "data_ret_v3_full_w2v"
MODEL_PATH_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"


# =========================
# Parsing helpers
# =========================

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)
VALUE_OF_RE = re.compile(r"^the value of (.+?) in .+? document in \d{4}\??$", re.IGNORECASE)


def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    q = (question or "").strip()
    m = DOC_YEAR_RE.search(q)
    if not m:
        return None, None
    doc = m.group(1).strip().lower()
    year = int(m.group(2))
    return doc, year


def extract_metric(question: str) -> Optional[str]:
    q = (question or "").strip()
    m = VALUE_OF_RE.match(q)
    if not m:
        return None
    return m.group(1).strip()


# =========================
# Chroma helpers
# =========================

def build_collection(chroma_dir: Path, collection_name: str, model_path: Path) -> Any:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = Word2VecEmbeddingFunction(model_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_where_filter(doc: Optional[str], year: Optional[int]) -> Optional[Dict[str, Any]]:
    if doc is None or year is None:
        return None
    # Chroma requires exactly 1 operator per level -> we use $and
    return {"$and": [{"doc": {"$eq": doc}}, {"year": {"$eq": int(year)}}]}


def retrieve(
    collection: Any,
    question: str,
    top_k: int,
    where_filter: Optional[Dict[str, Any]],
    retrieval_query_text: Optional[str] = None,
) -> Dict[str, Any]:
    # Improves retrieval on tables: use "metric" as query text
    qt = retrieval_query_text.strip() if retrieval_query_text else question

    kwargs = dict(
        query_texts=[qt],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    res = collection.query(**kwargs)
    return res


# =========================
# Gemini helpers
# =========================

def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not defined in the environment."
        )
    return key


def call_gemini(
    prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Tuple[str, Any, Any]:
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    text = getattr(resp, "text", None)
    if not text:
        try:
            cand = resp.candidates[0]
            parts = cand.content.parts
            text = "".join([p.text for p in parts if getattr(p, "text", None)]).strip()
        except Exception:
            text = str(resp).strip()

    finish = None
    usage = None
    try:
        finish = resp.candidates[0].finish_reason
    except Exception:
        pass
    try:
        usage = resp.usage_metadata
    except Exception:
        pass

    return text, finish, usage


# =========================
# Prompt builder (FINAL ROBUST VERSION)
# =========================

def build_prompt(question: str, contexts: List[str], metadatas: List[Dict], answer_style: str) -> str:
    
    # Identify the target year to reinforce in the prompt
    target_doc, target_year = parse_doc_year(question)
    year_instruction = ""
    if target_year:
        year_instruction = f"THE TARGET YEAR IS {target_year}. FOCUS ON COLUMNS OR DATA LABELED {target_year}.\n"

    base_rules = (
        "You are an intelligent data analyst. You are reading fragments of a financial/sustainability report.\n"
        "Your goal is to extract a specific value based on the user question.\n\n"
        "IMPORTANT RULES FOR FRAGMENTED TABLES:\n"
        "1. The document text is split into chunks. Headers (like '2022 | 2021 | 2020') might be in one chunk, and the data rows in another.\n"
        "2. Look for patterns. If you see a header row in one Context defining the column order, APPLY that order to data rows in other Contexts.\n"
        "3. Common column orders are descending (Current Year, Previous Year...) or ascending.\n"
        f"{year_instruction}"
        "4. If the question asks for a sum (e.g., 'Total Scope 1 and 2'), and you see the individual components, sum them up yourself.\n"
    )

    if answer_style == "value":
        style_instr = (
            "Output Format: Return ONLY the numeric value.\n"
            "Do not write sentences. Do not write 'I don't know' unless the data is completely missing.\n"
            "If you are 70% sure based on the table structure, output the value.\n"
        )
    elif answer_style == "free":
        style_instr = (
            "Output Format: Answer naturally. Explain HOW you found the value.\n"
            "Example: 'Based on Context 1, the table header order is 2021|2020. In Context 3, the row for GHG Emissions has value X in the first column, so the answer is X.'\n"
        )
    else:
        style_instr = "Answer naturally."

    ctx_block = []
    for i, (txt, meta) in enumerate(zip(contexts, metadatas), 1):
        doc_year = meta.get("year", "Unknown")
        doc_name = meta.get("doc", "Unknown")
        # Aggressive metadata injection
        header = f"--- [Context {i} | Document: {doc_name} | Metadata Year: {doc_year}] ---"
        ctx_block.append(f"{header}\n{txt}\n")

    return (
        base_rules
        + "\n" + style_instr
        + "\n=== CONTEXTS ===\n"
        + "\n".join(ctx_block)
        + "\n=== END CONTEXTS ===\n\n"
        + "Question: " + question
        + "\nAnswer:"
    )


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH_DEFAULT))
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT))

    ap.add_argument("--q", default=None, help="Free text question (string)")
    ap.add_argument("--i", type=int, default=None, help="Index from CSV to use")
    ap.add_argument("--auto-find", type=int, default=None, help="Automatically pick an index")

    # K=15 allows the model to see headers far from the data rows
    ap.add_argument("--k", type=int, default=15)
    
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")

    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temp", type=float, default=0.1)
    
    # 8192 is the max output for Gemini Flash (virtually unlimited for text)
    ap.add_argument("--max-out", type=int, default=8192)

    args = ap.parse_args()

    # Setup
    csv_path = Path(args.csv)
    chroma_dir = Path(args.chroma_dir)
    model_path = Path(args.model_path)
    collection = build_collection(chroma_dir, args.collection, model_path)
    df = pd.read_csv(csv_path)

    # Determine question
    if args.q:
        question = args.q.strip()
    elif args.i is not None:
        idx = int(args.i)
        question = str(df.loc[idx, "Question"])
    elif args.auto_find is not None:
        idx = int(args.auto_find)
        question = str(df.loc[idx, "Question"])
    else:
        print("Error: Use --q, --i or --auto-find")
        return

    doc, year = parse_doc_year(question)
    where_filter = build_where_filter(doc, year)
    metric = extract_metric(question)
    
    # Retrieve
    res = retrieve(collection, question, args.k, where_filter, metric)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # Prompt & Call
    prompt = build_prompt(question, docs, metas, args.answer_style)
    answer, finish, usage = call_gemini(prompt, args.model, args.temp, args.max_out)

    # Output
    if args.mode == "chat":
        print(answer)
    else:
        # Detailed debug
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Filter: {where_filter}")
        print(f"[DEBUG] Tokens (Input/Output): {usage.prompt_token_count}/{usage.candidates_token_count} (approx)")
        print(f"\n[DEBUG] PROMPT SENT:\n{'-'*40}\n{prompt}\n{'-'*40}\n")
        print(f"[DEBUG] GEMINI ANSWER:\n{answer}")

if __name__ == "__main__":
    main()
