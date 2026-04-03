"""
07_rag_agent_gemini.py
- Agentic RAG implementation using Gemini Function Calling natively.
- The Gemini Agent can autonomously decide to query the database multiple times
  to answer complex, multi-hop questions.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from google import genai
from google.genai import types


# =========================
# Paths / Config defaults
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR_DEFAULT = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "chroma" if LOCALAPPDATA else (PROJECT_ROOT / "data" / "chroma")
)
COLLECTION_DEFAULT = "data_ret_contexts_v2_chunks_meta"
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

# Global references for the Tool function
_COLLECTION = None
_DEBUG_MODE = False


# =========================
# Chroma DB initialization
# =========================

def init_collection(chroma_dir: Path, collection_name: str, embedding_model: str) -> Any:
    global _COLLECTION
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
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

    # Build where filter
    where_conditions = []
    if doc_name and doc_name.strip():
        where_conditions.append({"doc": {"$eq": doc_name.strip().lower()}})
        
    # We purposefully DO NOT stringently filter by `year` in ChromaDB
    # because table headers might be tagged with a different year.
    # Instead, we inject the year into the querying string to prioritize it semantically.
    if year and year > 1900:
        query = f"{query} in year {year} or table headers"

    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    if _DEBUG_MODE:
        print(f"\n[DEBUG Tool Execution] Tool 'search_database' called by Agent:")
        print(f"  -> query: '{query}'")
        print(f"  -> doc_name: '{doc_name}'")
        print(f"  -> year: {year}")
        print(f"  -> computed filter: {where_filter}")

    # Query Chroma
    kwargs = dict(
        query_texts=[query],
        n_results=10,  # We give it top 10 chunks per query
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
                header = f"--- [Context {i} | Document: {m_doc} | Metadata Year: {m_year}] ---"
                ctx_block.append(f"{header}\n{txt}\n")
            result_str = "\n".join(ctx_block)
            
        if _DEBUG_MODE:
            print(f"[DEBUG Tool Execution] Retrieved {len(docs)} chunks.")
            
        return result_str
        
    except Exception as e:
        error_msg = f"Error during database search: {str(e)}"
        if _DEBUG_MODE:
            print(f"[DEBUG Tool Execution] {error_msg}")
        return error_msg


# =========================
# System prompt & Agent initialization
# =========================

def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not defined in the environment."
        )
    return key


def run_agent(question: str, model_name: str, temperature: float, answer_style: str) -> str:
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    if answer_style == "value":
        behavior_instr = (
            "You must return ONLY the exact numeric value (or extremely short answer) without any additional text or formatting. "
            "Do not write sentences. If the user asks for multiple numbers, you can output them, but keep it minimal."
        )
    elif answer_style == "free":
        behavior_instr = (
            "Explain HOW you found the value naturally. Be detailed about which years/documents you analyzed."
        )
    else:
        behavior_instr = "Answer naturally and concisely based on the context."

    system_instruction = (
        "You are an intelligent data analyst Agent navigating financial and sustainability reports.\n"
        "You have access to a semantic search tool that queries a database of report fragments.\n"
        "Rules:\n"
        "1. Always use the 'search_database' tool to find the required data.\n"
        "2. If you need comparative data (e.g. year 2021 vs 2020), call the tool multiple times (once for 2021, once for 2020) if needed.\n"
        "3. Fragmented tables exist. You may need to cross-reference multiple chunks. Look for table headers (e.g. '2022 | 2021 | 2020') in one chunk to understand the column order for data rows in another chunk.\n"
        "4. If a chunk contains values separated by pipes like '12 | 14 | 16' and you don't know the years, assume standard descending order (Current year first) or search for another chunk containing the headers.\n"
        "5. If the question asks for a sum ('Total Scope 1 and 2'), retrieve both individually and sum them up yourself.\n"
        f"6. {behavior_instr}"
    )

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        tools=[search_database]
    )

    # Initialize a chat session so it maintains state of tool calls automatically
    chat = client.chats.create(
        model=model_name,
        config=config,
    )

    if _DEBUG_MODE:
        print("\n[DEBUG] Sending Question to Agent...")
        
    response = chat.send_message(question)
    
    # The SDK automatically handles the back-and-forth of tool calls unless we disable automatic function calling.
    return response.text


# =========================
# Main
# =========================

def main():
    global _DEBUG_MODE
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)

    ap.add_argument("--q", required=True, help="Question for the Agent")
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")

    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temp", type=float, default=0.1)

    args = ap.parse_args()

    # Set debug mode
    if args.mode == "debug":
        _DEBUG_MODE = True

    # Setup DB
    init_collection(Path(args.chroma_dir), args.collection, args.embedding_model)

    try:
        answer = run_agent(args.q, args.model, args.temp, args.answer_style)
        
        # When printing to stdout, just print the final answer (unless debugging, then all the tool call logs have already printed)
        if _DEBUG_MODE:
            print(f"\n[DEBUG] FINAL AGENT ANSWER:\n{'-'*40}\n{answer}\n{'-'*40}\n")
        else:
            # We strictly print ONLY the answer to avoid corrupting UI responses
            sys.stdout.write(answer + "\n")
            
    except Exception as e:
        sys.stderr.write(f"Agent Execution Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
