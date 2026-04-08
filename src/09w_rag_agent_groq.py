"""
09w_rag_agent_groq.py (Word2Vec version)
- Agentic RAG implementation using Groq API (Llama-3.1-70b-versatile).
- The Groq Agent autonomously calls tools to query the database.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

import chromadb

from groq import Groq
import openai
import time

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
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not input:
            return []
        
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings_array = self.svd.transform(tfidf_vecs)
        embeddings = [list(row) for row in embeddings_array]
        return embeddings
    
    def embed_query(self, input: str) -> list[float]:
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
LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR_DEFAULT = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_w2v" if LOCALAPPDATA else (PROJECT_ROOT / "data" / "chroma_w2v")
)
COLLECTION_DEFAULT = "data_ret_v3_full_w2v"
MODEL_PATH_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

_COLLECTION = None
_DEBUG_MODE = False

# =========================
# Chroma DB initialization
# =========================

def init_collection(chroma_dir: Path, collection_name: str, model_path: Path) -> Any:
    global _COLLECTION
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = Word2VecEmbeddingFunction(model_path)
    _COLLECTION = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _COLLECTION


# =========================
# Agent Tools (Function Calling)
# =========================

def search_database(query: str, doc_name: str = "", year: int = 0) -> str:
    """Searches the vector database for fragments of financial/sustainability reports to answer questions.
    Call this tool whenever you need facts, numbers, or information from the company's reports.
    If you need data for multiple years or companies, you can call this tool multiple times.

    Args:
        query: The semantic search query. Use specific keywords (e.g. 'Scope 1 emissions', 'total revenue').
        doc_name: (Optional) The name of the document or company. Leave empty if unknown.
        year: (Optional) The year of the report (e.g. 2021). Pass 0 if unknown.

    Returns:
        A formatted string containing the retrieved chunks of text from the reports.
    """
    global _COLLECTION, _DEBUG_MODE
    if _COLLECTION is None:
        return "Error: Database collection not initialized."

    where_conditions = []
    if doc_name and str(doc_name).strip():
        where_conditions.append({"doc": {"$eq": str(doc_name).strip().lower()}})
        
    # We purposefully DO NOT filter by `year` in ChromaDB so we don't lose headers
    # and we definitely do NOT mangle the query text, which destroys embedding density!


    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    if _DEBUG_MODE:
        print(f"\n[DEBUG Tool Execution] Tool 'search_database' called:")
        print(f"  -> query: '{query}'")
        print(f"  -> doc_name: '{doc_name}'")

    kwargs = dict(
        query_texts=[query],
        n_results=12, 
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    try:
        res = _COLLECTION.query(**kwargs)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        
        if not docs:
            result_str = "No documents found matching the search criteria."
        else:
            ctx_block = []
            for i, (txt, meta) in enumerate(zip(docs, metas), 1):
                m_year = meta.get("year", "Unknown")
                m_doc = meta.get("doc", "Unknown")
                # Removed truncation to give the model full context
                header = f"--- [Context {i} | Document: {m_doc} | Metadata Year: {m_year}] ---"
                ctx_block.append(f"{header}\n{txt}\n")
            result_str = "\n".join(ctx_block)
            
        return result_str
        
    except Exception as e:
        return f"Error during database search: {str(e)}"

# Define the JSON schema for the tool so Groq knows how to use it
DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Searches the vector database for fragments of financial/sustainability reports to answer questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The semantic search query. Ex: 'Scope 1 emissions' or 'total revenue'."
                },
                "doc_name": {
                    "type": "string",
                    "description": "The name of the document, company, or ticker (e.g. 'Oceana', 'Absa', 'AAPL', 'V'). Leave empty if unknown."
                },
                "year": {
                    "type": "integer",
                    "description": "The year of the report (e.g. 2021). Pass 0 if unknown."
                }
            },
            "required": ["query"]
        }
    }
}

# =========================
# Groq Agent Logic
# =========================

def get_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not defined in the environment. Please get one from console.groq.com"
        )
    return key

def run_agent(question: str, model_name: str, temperature: float, answer_style: str) -> tuple[str, int]:
    groq_api_key = get_api_key()
    gemini_api_key = os.getenv("GEMINI_API_KEY")


    # Build prompt instructions
    if answer_style == "value":
        behavior_instr = (
            "You must return ONLY the exact numeric value without any additional text. "
            "Do not write sentences. Keep it incredibly minimal, just the extracted numbers."
        )
    elif answer_style == "free":
        behavior_instr = "Explain HOW you found the value naturally. Be detailed about which years/documents you analyzed."
    else:
        behavior_instr = "Answer naturally and concisely based on the context."

    system_instruction = (
        "You are an expert financial data extraction bot. Your ONLY job is to find a specific numeric value from company reports stored in a vector database.\n\n"
        "CRITICAL RULES — VIOLATING ANY OF THESE IS FAILURE:\n"
        "1. ALWAYS call search_database. NEVER say 'I don't have access' — you DO have access via the tool.\n"
        "2. NEVER ask the user to search again or suggest they do anything. YOU must keep searching until you find it.\n"
        "3. NEVER reply with sentences when answer_style is 'value'. Reply with ONLY the number.\n"
        "4. If your first search returns no useful result, TRY AGAIN with different keywords or without the doc_name filter.\n"
        "5. Tables in chunks use '|' as column separators. The header row shows years (e.g. 2021 | 2020 | 2019). Match the requested year to the correct column CAREFULLY.\n"
        "6. Many reports contain historical data for multiple years. A 2023 report WILL have 2021 and 2020 data in its tables.\n"
        "7. If you see the metric but cannot determine the correct year column, return your best estimate rather than giving up.\n"
        "8. For sums (e.g. 'Total Scope 1 and 2'), search each component and compute the sum yourself.\n"
        f"9. {behavior_instr}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question}
    ]

    # The default prefix if the user passed standard name
    if not model_name.startswith("groq:") and not model_name.startswith("gemini:"):
        model_name = f"groq:{model_name}"

    models_queue = [model_name] + [m for m in [
        "groq:llama-3.3-70b-versatile",
        "gemini:gemini-2.5-flash",
        "gemini:gemini-2.0-flash",
        "gemini:gemini-2.0-flash-lite-001",
        "groq:meta-llama/llama-4-scout-17b-16e-instruct",
        "groq:llama-3.1-8b-instant",
        "groq:qwen/qwen3-32b"
    ] if m != model_name]

    # Agent Loop
    MAX_STEPS = 10
    num_searches = 0
    for step in range(MAX_STEPS):
        response = None
        attempts_this_step = 0
        while models_queue:
            target_model = models_queue[0]
            provider, actual_model_name = target_model.split(":", 1)
            
            if provider == "groq":
                client = Groq(api_key=groq_api_key, max_retries=0)
            else:
                if not gemini_api_key:
                    if _DEBUG_MODE:
                        print(f"\n[DEBUG] Skipping {target_model} because GEMINI_API_KEY is not set.")
                    models_queue.pop(0)
                    continue
                client = openai.OpenAI(
                    api_key=gemini_api_key, 
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=0
                )
                
            try:
                response = client.chat.completions.create(
                    model=actual_model_name,
                    messages=messages,
                    tools=[DATABASE_TOOL],
                    tool_choice="auto",
                    temperature=temperature
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str
                
                if is_rate_limit and attempts_this_step < 2:
                    # Wait and retry the SAME model before giving up on it
                    attempts_this_step += 1
                    if _DEBUG_MODE:
                        print(f"\n[WARNING] Rate limit on {target_model}, waiting 15s (attempt {attempts_this_step}/2)...")
                    time.sleep(15)
                    continue
                
                failed_model = models_queue.pop(0)
                attempts_this_step = 0
                if _DEBUG_MODE:
                    print(f"\n[WARNING] Model {failed_model} failed ({err_str[:80]}...). Switching to {models_queue[0] if models_queue else 'None'}...")
                time.sleep(2)
                    
        if not models_queue or response is None:
            return "API ERROR: Rate limit exceeded for all available fallback models.", num_searches

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content, num_searches

        # Serialize to a plain dict — works uniformly for both Groq and Gemini clients
        assistant_msg: dict = {"role": "assistant", "content": response_message.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        for tool_call in tool_calls:
            if tool_call.function.name == "search_database":
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}
                
                tool_resp = search_database(
                    query=args.get("query", ""),
                    doc_name=args.get("doc_name", ""),
                    year=args.get("year", 0)
                )
                num_searches += 1

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_resp
                })

    return "Agent exhausted maximum execution steps without formulating an answer.", num_searches


# =========================
# Main
# =========================

def main():
    global _DEBUG_MODE
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT))

    ap.add_argument("--q", required=True, help="Question for the Agent")
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")

    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--temp", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "debug":
        _DEBUG_MODE = True

    init_collection(Path(args.chroma_dir), args.collection, Path(args.model_path))

    try:
        answer, num_searches = run_agent(args.q, args.model, args.temp, args.answer_style)
        
        if _DEBUG_MODE:
            print(f"\n[DEBUG] TOTAL SEARCHES: {num_searches}")
            print(f"\n[DEBUG] FINAL AGENT ANSWER:\n{'-'*40}\n{answer}\n{'-'*40}\n")
        else:
            sys.stdout.write(answer + "\n")
            
    except Exception as e:
        sys.stderr.write(f"Agent Execution Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
