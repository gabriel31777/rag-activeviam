"""
04_rag_agent.py
Agent RAG avec fallback multi-modèle (Groq/Gemini).

Source de données : PDFs indexés dans ChromaDB (PAS le CSV).

Supporte trois méthodes d'embedding :
    --embedding tfidf_svd          : TF-IDF + SVD
    --embedding word2vec           : Word2Vec (gensim)
    --embedding sentence_transformer : SentenceTransformers

Utilisation :
    python src/04_rag_agent.py --embedding tfidf_svd --q "What is the Scope 1 in Sasol 2021?"
    python src/04_rag_agent.py --embedding word2vec --q "Total revenue in Clicks 2020?" --mode debug
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import chromadb
from groq import Groq
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import get_embedding_function


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALAPPDATA = os.environ.get("LOCALAPPDATA", ".")
MODELS_DIR = Path(LOCALAPPDATA) / "rag-activeviam" / "models"

# Configuration par type d'embedding (même que 02_index_pdfs.py)
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

DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)

# Variables globales
_COLLECTION = None
_COLLECTION_TFIDF = None
_COLLECTION_ST = None
_IS_HYBRID = False
_DEBUG_MODE = False


# =========================
# Initialisation ChromaDB
# =========================

def init_collection(embedding_type: str = "tfidf_svd") -> Any:
    """Initialise la collection ChromaDB selon le type d'embedding."""
    global _COLLECTION, _COLLECTION_TFIDF, _COLLECTION_ST, _IS_HYBRID

    if embedding_type == "hybrid":
        _IS_HYBRID = True
        # Init TF-IDF
        cfg_tfidf = EMBEDDING_CONFIG["tfidf_svd"]
        emb_fn_tfidf = get_embedding_function("tfidf_svd", model_path=str(cfg_tfidf["model_path"]))
        client_tfidf = chromadb.PersistentClient(path=str(cfg_tfidf["chroma_dir"]))
        _COLLECTION_TFIDF = client_tfidf.get_or_create_collection(
            name=cfg_tfidf["collection"], embedding_function=emb_fn_tfidf
        )
        
        # Init ST
        cfg_st = EMBEDDING_CONFIG["sentence_transformer"]
        emb_fn_st = get_embedding_function("sentence_transformer")
        client_st = chromadb.PersistentClient(path=str(cfg_st["chroma_dir"]))
        _COLLECTION_ST = client_st.get_or_create_collection(
            name=cfg_st["collection"], embedding_function=emb_fn_st
        )
        return (_COLLECTION_TFIDF, _COLLECTION_ST)
    else:
        _IS_HYBRID = False
        config = EMBEDDING_CONFIG[embedding_type]
        chroma_dir = config["chroma_dir"]
        collection_name = config["collection"]
        model_path = config["model_path"]

        emb_fn = get_embedding_function(
            embedding_type,
            model_path=str(model_path) if model_path else None,
        )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        _COLLECTION = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return _COLLECTION


# =========================
# Outil de l'agent (function calling)
# =========================

def search_database(query: str, doc_name: str = "", year: int = 0) -> str:
    """Recherche dans la base de données vectorielle des fragments de rapports financiers/ESG."""
    global _COLLECTION, _COLLECTION_TFIDF, _COLLECTION_ST, _IS_HYBRID, _DEBUG_MODE
    
    if not _IS_HYBRID and _COLLECTION is None:
        return "Erreur : la collection n'est pas initialisée."
    if _IS_HYBRID and (_COLLECTION_TFIDF is None or _COLLECTION_ST is None):
        return "Erreur : les collections pour le mode hybride ne sont pas initialisées."

    where_conditions = []
    if doc_name and str(doc_name).strip():
        where_conditions.append({"doc": {"$eq": str(doc_name).strip().lower()}})

    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    if _DEBUG_MODE:
        print(f"\n[DEBUG Outil] 'search_database' appelé :")
        print(f"  -> requête : '{query}'")
        print(f"  -> document : '{doc_name}'")

    kwargs = dict(
        query_texts=[query],
        n_results=12,
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    try:
        if _IS_HYBRID:
            res1 = _COLLECTION_TFIDF.query(**kwargs)
            res2 = _COLLECTION_ST.query(**kwargs)
            
            scores = {}
            doc_map = {}
            meta_map = {}
            
            for rank, (txt, meta) in enumerate(zip(res1["documents"][0], res1["metadatas"][0])):
                uid = f"{meta.get('doc', '')}_{meta.get('page', '')}"
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                meta_map[uid] = meta
                
            for rank, (txt, meta) in enumerate(zip(res2["documents"][0], res2["metadatas"][0])):
                uid = f"{meta.get('doc', '')}_{meta.get('page', '')}"
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                meta_map[uid] = meta
                
            sorted_uids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            docs = [doc_map[uid] for uid in sorted_uids[:12]]
            metas = [meta_map[uid] for uid in sorted_uids[:12]]
        else:
            res = _COLLECTION.query(**kwargs)
            docs = res["documents"][0]
            metas = res["metadatas"][0]

        if not docs:
            return "Aucun document trouvé correspondant aux critères."

        ctx_block = []
        for i, (txt, meta) in enumerate(zip(docs, metas), 1):
            m_year = meta.get("year", "Inconnu")
            m_doc = meta.get("doc", "Inconnu")
            m_page = meta.get("page", "?")
            header = f"--- [Contexte {i} | Document : {m_doc} | Année : {m_year} | Page : {m_page}] ---"
            ctx_block.append(f"{header}\n{txt}\n")
        return "\n".join(ctx_block)

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"


# Schéma JSON de l'outil
DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Recherche dans la base de données vectorielle des fragments de rapports financiers/ESG.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requête de recherche sémantique (ex: 'Scope 1 emissions', 'total revenue').",
                },
                "doc_name": {
                    "type": "string",
                    "description": "Nom du document ou de l'entreprise (ex: 'Oceana', 'Absa'). Vide si inconnu.",
                },
                "year": {
                    "type": "integer",
                    "description": "Année du rapport (ex: 2021). 0 si inconnue.",
                },
            },
            "required": ["query"],
        },
    },
}


# =========================
# Logique de l'agent
# =========================

def get_api_key() -> str:
    """Récupère la clé API Groq."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY non définie.")
    return key


def run_agent(
    question: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    answer_style: str = "value",
) -> tuple[str, int]:
    """Exécute l'agent RAG avec fallback multi-modèle.

    Retourne (réponse, nombre_de_recherches).
    """
    groq_api_key = get_api_key()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if answer_style == "value":
        behavior_instr = (
            "Vous devez retourner UNIQUEMENT la valeur numérique exacte sans texte supplémentaire. "
            "N'écrivez pas de phrases. Gardez la réponse minimale."
        )
    elif answer_style == "free":
        behavior_instr = (
            "Expliquez COMMENT vous avez trouvé la valeur. Détaillez les années/documents analysés."
        )
    else:
        behavior_instr = "Répondez naturellement et de façon concise."

    system_instruction = (
        "Vous êtes un bot expert en extraction de données financières. Votre SEUL travail est de trouver "
        "une valeur numérique spécifique dans des rapports d'entreprise stockés dans une base vectorielle.\n\n"
        "RÈGLES CRITIQUES :\n"
        "1. Appelez TOUJOURS search_database. Ne dites JAMAIS 'je n'ai pas accès' — vous AVEZ accès via l'outil.\n"
        "2. Ne demandez JAMAIS à l'utilisateur de chercher. VOUS devez continuer à chercher jusqu'à trouver.\n"
        "3. Ne répondez JAMAIS avec des phrases quand answer_style est 'value'. Répondez UNIQUEMENT avec le nombre.\n"
        "4. Si la première recherche ne donne rien d'utile, RÉESSAYEZ avec d'autres mots-clés ou sans filtre doc_name.\n"
        "5. Les tableaux utilisent '|' comme séparateur. L'en-tête montre les années (ex: 2021 | 2020 | 2019). "
        "Associez l'année demandée à la bonne colonne SOIGNEUSEMENT.\n"
        "6. Les rapports peuvent contenir des données historiques pour plusieurs années.\n"
        "7. Si vous voyez la métrique mais ne pouvez pas identifier la colonne, donnez votre meilleure estimation.\n"
        "8. Pour les sommes (ex: 'Total Scope 1 et 2'), cherchez chaque composante et calculez la somme.\n"
        f"9. {behavior_instr}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]

    if not model_name.startswith("groq:") and not model_name.startswith("gemini:"):
        model_name = f"groq:{model_name}"

    all_models = [model_name] + [
        m for m in [
            "groq:llama-3.3-70b-versatile",
            "groq:meta-llama/llama-4-scout-17b-16e-instruct",
            "groq:llama-3.1-8b-instant",
            "groq:qwen/qwen3-32b",
            "gemini:gemini-3-flash-preview",
            "gemini:gemini-2.5-pro",
            "gemini:gemini-2.5-flash",
        ]
        if m != model_name
    ]

    MAX_STEPS = 5
    MAX_SEARCHES = 10
    num_searches = 0
    last_client = None
    last_model_name = None

    for step in range(MAX_STEPS):
        if num_searches >= MAX_SEARCHES:
            messages.append({
                "role": "user",
                "content": (
                    "Vous avez atteint la limite de recherches. "
                    "Répondez MAINTENANT avec les informations déjà trouvées. "
                    "Si vous n'avez pas trouvé la valeur exacte, donnez votre meilleure estimation."
                ),
            })
            if last_client and last_model_name:
                try:
                    final_resp = last_client.chat.completions.create(
                        model=last_model_name,
                        messages=messages,
                        temperature=temperature,
                    )
                    return final_resp.choices[0].message.content, num_searches
                except Exception:
                    pass
            return "Limite de recherches atteinte.", num_searches

        response = None
        while True:
            models_queue = list(all_models)
            
            while models_queue:
                target_model = models_queue[0]
                provider, actual_model_name = target_model.split(":", 1)

                if provider == "groq":
                    client = Groq(api_key=groq_api_key, max_retries=0)
                else:
                    if not gemini_api_key:
                        models_queue.pop(0)
                        continue
                    client = openai.OpenAI(
                        api_key=gemini_api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        max_retries=0,
                    )

                try:
                    response = client.chat.completions.create(
                        model=actual_model_name,
                        messages=messages,
                        tools=[DATABASE_TOOL],
                        tool_choice="auto",
                        temperature=temperature,
                    )
                    last_client = client
                    last_model_name = actual_model_name
                    break
                except Exception as e:
                    failed_model = models_queue.pop(0)
                    if _DEBUG_MODE:
                        err_msg = str(e).split('\n')[0][:100]
                        print(f"\n[ATTENTION] {failed_model} en échec ({err_msg}...). Bascule...")
                    time.sleep(2)
                    
            if response is not None:
                break
                
            if _DEBUG_MODE:
                print("\n[ATTENTION] Tous les modèles sont épuisés (Rate limits, etc.). Attente de 60s avant de tout réessayer...")
            else:
                sys.stdout.write("\n[ATTENTION] Limite d'API atteinte. Attente de 60s...\n")
                sys.stdout.flush()
                
            time.sleep(60)

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content, num_searches

        assistant_msg = {"role": "assistant", "content": response_message.content or ""}
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
                    year=args.get("year", 0),
                )
                num_searches += 1

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_resp,
                })

    return "L'agent a épuisé le nombre maximum d'étapes.", num_searches


# =========================
# Point d'entrée
# =========================

def main():
    global _DEBUG_MODE

    ap = argparse.ArgumentParser(description="Agent RAG (Groq/Gemini)")
    ap.add_argument(
        "--embedding",
        choices=["tfidf_svd", "word2vec", "sentence_transformer", "hybrid"],
        default="tfidf_svd",
    )
    ap.add_argument("--q", required=True, help="Question pour l'agent")
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--temp", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "debug":
        _DEBUG_MODE = True

    init_collection(args.embedding)

    try:
        answer, num_searches = run_agent(args.q, args.model, args.temp, args.answer_style)

        if _DEBUG_MODE:
            print(f"\n[DEBUG] RECHERCHES : {num_searches}")
            print(f"\n[DEBUG] RÉPONSE :\n{'-'*40}\n{answer}\n{'-'*40}\n")
        else:
            sys.stdout.write(answer + "\n")

    except Exception as e:
        sys.stderr.write(f"Erreur : {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
