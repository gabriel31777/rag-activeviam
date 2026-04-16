"""
04_rag_agent.py
Agent RAG avec fallback multi-modele (Groq/Gemini).
Recherche dans les PDFs indexes dans ChromaDB.

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


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALAPPDATA = os.environ.get("LOCALAPPDATA", ".")
MODELS_DIR = Path(LOCALAPPDATA) / "rag-activeviam" / "models"

# Config par type d'embedding (meme que 02_index_pdfs.py)
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


_COLLECTION = None
_COLLECTION_TFIDF = None
_COLLECTION_ST = None
_IS_HYBRID = False
_DEBUG_MODE = False

def _log(msg: str) -> None:
    """Log sur stderr."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# Initialisation ChromaDB

def init_collection(embedding_type: str = "tfidf_svd") -> Any:
    """Charge la collection ChromaDB pour le type d'embedding choisi."""
    global _COLLECTION, _COLLECTION_TFIDF, _COLLECTION_ST, _IS_HYBRID

    _log(f"[1/4] INITIALISATION -- Embedding: {embedding_type}")

    if embedding_type == "hybrid":
        _IS_HYBRID = True

        cfg_tfidf = EMBEDDING_CONFIG["tfidf_svd"]
        _log("  -> Chargement du modele TF-IDF + SVD...")
        emb_fn_tfidf = get_embedding_function("tfidf_svd", model_path=str(cfg_tfidf["model_path"]))
        client_tfidf = chromadb.PersistentClient(path=str(cfg_tfidf["chroma_dir"]))
        _COLLECTION_TFIDF = client_tfidf.get_or_create_collection(
            name=cfg_tfidf["collection"], embedding_function=emb_fn_tfidf
        )
        _log(f"  -> Collection TF-IDF chargee ({_COLLECTION_TFIDF.count()} chunks)")
        

        cfg_st = EMBEDDING_CONFIG["sentence_transformer"]
        _log("  -> Chargement du modele SentenceTransformers...")
        emb_fn_st = get_embedding_function("sentence_transformer")
        client_st = chromadb.PersistentClient(path=str(cfg_st["chroma_dir"]))
        _COLLECTION_ST = client_st.get_or_create_collection(
            name=cfg_st["collection"], embedding_function=emb_fn_st
        )
        _log(f"  -> Collection SentenceTransformers chargee ({_COLLECTION_ST.count()} chunks)")
        _log("  -> Mode HYBRIDE actif (RRF: TF-IDF + SentenceTransformers)")
        return (_COLLECTION_TFIDF, _COLLECTION_ST)
    else:
        _IS_HYBRID = False
        config = EMBEDDING_CONFIG[embedding_type]
        chroma_dir = config["chroma_dir"]
        collection_name = config["collection"]
        model_path = config["model_path"]

        _log(f"  -> Chargement du modele {embedding_type}...")
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
        _log(f"  -> Collection '{collection_name}' chargee ({_COLLECTION.count()} chunks)")
        return _COLLECTION


# Outil de recherche (function calling)

def search_database(query: str, doc_name: str = "", year: int = 0) -> str:
    """Recherche dans la base vectorielle les fragments pertinents."""
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

    filter_info = f"doc='{doc_name}'" if doc_name else "sans filtre"
    _log(f"[3/4] RECHERCHE dans ChromaDB -- requete: '{query}' | {filter_info}")
    if _DEBUG_MODE:
        print(f"\n[DEBUG] search_database : query='{query}', doc='{doc_name}'")

    kwargs = dict(
        query_texts=[query],
        n_results=12,
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    try:
        if _IS_HYBRID:
            _log("  -> Recherche parallele: TF-IDF + SentenceTransformers")
            res1 = _COLLECTION_TFIDF.query(**kwargs)
            res2 = _COLLECTION_ST.query(**kwargs)
            _log(f"  -> TF-IDF: {len(res1['documents'][0])} resultats | ST: {len(res2['documents'][0])} resultats")
            
            scores = {}
            doc_map = {}
            meta_map = {}
            
            for rank, (txt, meta, idstr) in enumerate(zip(res1["documents"][0], res1["metadatas"][0], res1["ids"][0])):
                uid = idstr
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                meta_map[uid] = meta
                
            for rank, (txt, meta, idstr) in enumerate(zip(res2["documents"][0], res2["metadatas"][0], res2["ids"][0])):
                uid = idstr
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (60 + rank)
                doc_map[uid] = txt
                meta_map[uid] = meta
                
            sorted_uids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            docs = [doc_map[uid] for uid in sorted_uids[:12]]
            metas = [meta_map[uid] for uid in sorted_uids[:12]]
            _log(f"  -> Fusion RRF: {len(docs)} fragments selectionnes")
        else:
            res = _COLLECTION.query(**kwargs)
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            _log(f"  -> {len(docs)} fragments recuperes")

        if not docs:
            return "Aucun document trouve correspondant aux criteres."

        ctx_block = []
        for i, (txt, meta) in enumerate(zip(docs, metas), 1):
            m_year = meta.get("year", "?")
            m_doc = meta.get("doc", "?")
            m_page = meta.get("page", "?")
            header = f"--- [Contexte {i} | Document : {m_doc} | Annee : {m_year} | Page : {m_page}] ---"
            ctx_block.append(f"{header}\n{txt}\n")
        return "\n".join(ctx_block)

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"


# Schema JSON de l'outil pour le function calling
DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Recherche dans la base vectorielle les fragments de rapports financiers/ESG.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Requete de recherche semantique (ex: 'Scope 1 emissions', 'total revenue').",
                },
                "doc_name": {
                    "type": "string",
                    "description": "Nom du document ou de l'entreprise (ex: 'Oceana', 'Absa'). Vide si inconnu.",
                },
                "year": {
                    "type": "integer",
                    "description": "Annee du rapport (ex: 2021). 0 si inconnue.",
                },
            },
            "required": ["query"],
        },
    },
}


# Logique de l'agent

def get_api_key() -> str:
    """Recupere la cle API Groq."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY non definie.")
    return key


def run_agent(
    question: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
    answer_style: str = "value",
) -> tuple[str, int]:
    """Execute l'agent RAG. Retourne (reponse, nb_recherches)."""
    _log(f"[2/4] AGENT RAG -- Question: '{question[:80]}...'" if len(question) > 80 else f"[2/4] AGENT RAG -- Question: '{question}'")
    _log(f"  -> Style de reponse: {answer_style}")
    groq_api_key = get_api_key()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if answer_style == "value":
        behavior_instr = (
            "Vous devez retourner UNIQUEMENT la valeur numerique exacte sans texte supplementaire. "
            "N'ecrivez pas de phrases. Gardez la reponse minimale."
        )
    elif answer_style == "free":
        behavior_instr = (
            "Expliquez COMMENT vous avez trouve la valeur. Detaillez les annees/documents analyses."
        )
    else:
        behavior_instr = "Repondez naturellement et de facon concise."

    system_instruction = (
        "Vous etes un bot expert en extraction de donnees financieres. Votre SEUL travail est de trouver "
        "une valeur numerique specifique dans des rapports d'entreprise stockes dans une base vectorielle.\n\n"
        "REGLES CRITIQUES :\n"
        "1. Appelez TOUJOURS search_database. Ne dites JAMAIS 'je n'ai pas acces' -- vous AVEZ acces via l'outil.\n"
        "2. Ne demandez JAMAIS a l'utilisateur de chercher. VOUS devez continuer a chercher jusqu'a trouver.\n"
        "3. Ne repondez JAMAIS avec des phrases quand answer_style est 'value'. Repondez UNIQUEMENT avec le nombre.\n"
        "4. Si la premiere recherche ne donne rien d'utile, REESSAYEZ avec d'autres mots-cles ou sans filtre doc_name.\n"
        "5. Les tableaux utilisent '|' comme separateur. L'en-tete montre les annees (ex: 2021 | 2020 | 2019). "
        "Associez l'annee demandee a la bonne colonne SOIGNEUSEMENT.\n"
        "6. Les rapports peuvent contenir des donnees historiques pour plusieurs annees.\n"
        "7. Si vous voyez la metrique mais ne pouvez pas identifier la colonne, donnez votre meilleure estimation.\n"
        "8. Si vous n'arrivez vraiment pas a trouver l'information dans les fichiers (meme apres recherche), PUIS vous pouvez utiliser vos connaissances generales pour repondre. Mais dans ce cas, vous DEVEZ OBLIGATOIREMENT commencer votre reponse par : 'Non trouve dans les fichiers : '.\n"
        "9. Pour les sommes (ex: 'Total Scope 1 et 2'), cherchez chaque composante et calculez la somme.\n"
        f"10. {behavior_instr}"
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
                    "Repondez MAINTENANT avec les informations deja trouvees. "
                    "Si vous n'avez pas trouve la valeur exacte, donnez votre meilleure estimation."
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

        _log(f"  -> Etape {step + 1}/{MAX_STEPS} (recherches: {num_searches}/{MAX_SEARCHES})")
        response = None
        while True:
            models_queue = list(all_models)
            
            while models_queue:
                target_model = models_queue[0]
                provider, actual_model_name = target_model.split(":", 1)

                _log(f"  -> Appel LLM: {target_model}")
                if _DEBUG_MODE:
                    sys.stdout.write(f"  -> {target_model}... ")
                    sys.stdout.flush()

                if provider == "groq":
                    client = Groq(api_key=groq_api_key, max_retries=0)
                else:
                    if not gemini_api_key:
                        if _DEBUG_MODE:
                            sys.stdout.write("Ignore (pas de cle Gemini)\n")
                            sys.stdout.flush()
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
                    _log(f"     OK -- Reponse recue de {actual_model_name}")
                    if _DEBUG_MODE:
                        sys.stdout.write("OK\n")
                        sys.stdout.flush()
                    last_client = client
                    last_model_name = actual_model_name
                    break
                except Exception as e:
                    last_err = str(e)
                    models_queue.pop(0)
                    err_msg = last_err.split('\n')[0][:80]
                    _log(f"     ECHEC ({err_msg}) -> passage au modele suivant")
                    if _DEBUG_MODE:
                        sys.stdout.write(f"Echec ({err_msg}...)\n")
                        sys.stdout.flush()
                    
            if response is not None:
                break
                
            if _DEBUG_MODE:
                print(f"\n[ATTENTION] Tous les modeles sont epuises (Derniere erreur : {last_err}).")
            
            return f"Erreur : Tous les modeles sont surcharges (API Rate Limit). Veuillez reessayer dans quelques secondes. (Detail: {last_err[:50]})", num_searches

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            _log(f"[4/4] REPONSE -- {num_searches} recherches effectuees")
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

                _log(f"  -> Recherche {num_searches + 1}")
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

    return "Nombre maximum d'etapes atteint.", num_searches


# Main

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

    _log("="*50)
    _log("          RAG-ACTIVEVIAM AGENT")
    _log("="*50)
    init_collection(args.embedding)

    try:
        answer, num_searches = run_agent(args.q, args.model, args.temp, args.answer_style)
        _log("="*50)

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
