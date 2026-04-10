"""
07w_rag_agent_gemini.py (version TF-IDF + SVD)
Agent RAG utilisant l'API Gemini (Google GenAI) avec appel de fonction automatique.

L'agent a acces a un outil 'search_database' qu'il utilise de facon autonome
pour interroger la base de donnees vectorielle ChromaDB et trouver les
informations financières/ESG demandees.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import chromadb
from google import genai
from google.genai import types

# Import du module d'embedding partage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import TfidfSvdEmbeddingFunction


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v"
COLLECTION_DEFAULT = "data_ret_contexts_v2_chunks_meta_w2v"
MODEL_PATH_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)

# Variables globales
_COLLECTION = None
_DEBUG_MODE = False


# =========================
# Initialisation ChromaDB
# =========================

def init_collection(chroma_dir: Path, collection_name: str, model_path: Path) -> Any:
    """Initialise la collection ChromaDB avec l'embedding TF-IDF + SVD."""
    global _COLLECTION
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = TfidfSvdEmbeddingFunction(model_path)
    _COLLECTION = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _COLLECTION


# =========================
# Outil de l'agent (function calling)
# =========================

def search_database(query: str, doc_name: str = "", year: int = 0) -> str:
    """Recherche dans la base de donnees vectorielle des fragments de rapports financiers/ESG.

    Appelez cet outil chaque fois que vous avez besoin de faits, chiffres ou informations
    provenant des rapports des entreprises. Vous pouvez l'appeler plusieurs fois
    pour des annees ou entreprises differentes.

    Args:
        query: Requete de recherche semantique (ex: 'Scope 1 emissions', 'chiffre d'affaires').
        doc_name: Nom du document ou de l'entreprise (optionnel, vide si inconnu).
        year: Annee du rapport (ex: 2021). 0 si inconnue.

    Returns:
        Texte formate contenant les chunks recuperes.
    """
    global _COLLECTION, _DEBUG_MODE
    if _COLLECTION is None:
        return "Erreur : la collection n'est pas initialisee."

    # Construire le filtre
    where_conditions = []
    if doc_name and doc_name.strip():
        where_conditions.append({"doc": {"$eq": doc_name.strip().lower()}})

    # On injecte l'annee dans la requete semantique plutot que de filtrer strictement
    if year and year > 1900:
        query = f"{query} in year {year} or table headers"

    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    if _DEBUG_MODE:
        print(f"\n[DEBUG Outil] 'search_database' appele :")
        print(f"  -> requete : '{query}'")
        print(f"  -> document : '{doc_name}'")
        print(f"  -> annee : {year}")
        print(f"  -> filtre : {where_filter}")

    kwargs = dict(
        query_texts=[query],
        n_results=10,
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    try:
        res = _COLLECTION.query(**kwargs)
        docs = res["documents"][0]
        metas = res["metadatas"][0]

        if not docs:
            result_str = "Aucun document trouve correspondant aux criteres de recherche."
        else:
            ctx_block = []
            for i, (txt, meta) in enumerate(zip(docs, metas), 1):
                m_year = meta.get("year", "Inconnu")
                m_doc = meta.get("doc", "Inconnu")
                header = f"--- [Contexte {i} | Document : {m_doc} | Annee : {m_year}] ---"
                ctx_block.append(f"{header}\n{txt}\n")
            result_str = "\n".join(ctx_block)

        if _DEBUG_MODE:
            print(f"[DEBUG Outil] {len(docs)} chunks recuperes.")

        return result_str

    except Exception as e:
        error_msg = f"Erreur lors de la recherche : {str(e)}"
        if _DEBUG_MODE:
            print(f"[DEBUG Outil] {error_msg}")
        return error_msg


# =========================
# Cle API et logique de l'agent
# =========================

def get_api_key() -> str:
    """Recupere la cle API Gemini."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY non definie dans l'environnement.")
    return key


def run_agent(question: str, model_name: str, temperature: float, answer_style: str) -> str:
    """Execute l'agent RAG Gemini avec appel de fonction automatique."""
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    if answer_style == "value":
        behavior_instr = (
            "Vous devez retourner UNIQUEMENT la valeur numerique exacte sans texte supplementaire. "
            "N'ecrivez pas de phrases. Gardez la reponse minimale."
        )
    elif answer_style == "free":
        behavior_instr = (
            "Expliquez COMMENT vous avez trouve la valeur. Detaillez quelles annees/documents vous avez analyses."
        )
    else:
        behavior_instr = "Repondez naturellement et de facon concise."

    system_instruction = (
        "Vous etes un agent analyste de donnees financieres et ESG.\n"
        "Vous avez acces a un outil de recherche semantique pour interroger une base de rapports.\n"
        "Regles :\n"
        "1. Utilisez toujours 'search_database' pour trouver les donnees.\n"
        "2. Pour des donnees comparatives (ex: 2021 vs 2020), appelez l'outil plusieurs fois si necessaire.\n"
        "3. Les tableaux fragmentes existent. Croisez plusieurs chunks pour trouver en-tetes et donnees.\n"
        "4. Si un chunk contient des valeurs separees par '|' sans en-tetes, supposez l'ordre decroissant.\n"
        "5. Pour les sommes ('Total Scope 1 et 2'), recuperez chaque composante et calculez la somme.\n"
        f"6. {behavior_instr}"
    )

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        tools=[search_database],
    )

    chat = client.chats.create(
        model=model_name,
        config=config,
    )

    if _DEBUG_MODE:
        print("\n[DEBUG] Envoi de la question a l'agent...")

    response = chat.send_message(question)
    return response.text


# =========================
# Point d'entree
# =========================

def main():
    global _DEBUG_MODE

    ap = argparse.ArgumentParser(description="Agent RAG Gemini (TF-IDF + SVD)")
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT))

    ap.add_argument("--q", required=True, help="Question pour l'agent")
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")

    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temp", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "debug":
        _DEBUG_MODE = True

    init_collection(Path(args.chroma_dir), args.collection, Path(args.model_path))

    try:
        answer = run_agent(args.q, args.model, args.temp, args.answer_style)

        if _DEBUG_MODE:
            print(f"\n[DEBUG] REPONSE FINALE DE L'AGENT :\n{'-'*40}\n{answer}\n{'-'*40}\n")
        else:
            sys.stdout.write(answer + "\n")

    except Exception as e:
        sys.stderr.write(f"Erreur de l'agent : {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
