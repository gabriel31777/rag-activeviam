"""
09w_rag_agent_groq.py (version TF-IDF + SVD)
Agent RAG utilisant l'API Groq (Llama) avec fallback multi-modele.

L'agent utilise l'appel de fonction (function calling) pour interroger
la base de donnees vectorielle ChromaDB de facon autonome. En cas de
rate limiting, il bascule automatiquement sur des modeles de secours
(Groq -> Gemini -> autres).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

import chromadb
from groq import Groq
import openai

# Import du module d'embedding partage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddings import TfidfSvdEmbeddingFunction


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR_DEFAULT = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "chroma_w2v"
    if LOCALAPPDATA
    else (PROJECT_ROOT / "data" / "chroma_w2v")
)
COLLECTION_DEFAULT = "data_ret_v3_full_w2v"
MODEL_PATH_DEFAULT = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"
    if LOCALAPPDATA
    else (PROJECT_ROOT / "models" / "word2vec_pdf.pkl")
)

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

    Args:
        query: Requete de recherche semantique (ex: 'Scope 1 emissions', 'chiffre d'affaires total').
        doc_name: Nom du document ou de l'entreprise (optionnel).
        year: Annee du rapport (ex: 2021). 0 si inconnue.

    Returns:
        Texte formate contenant les chunks recuperes.
    """
    global _COLLECTION, _DEBUG_MODE
    if _COLLECTION is None:
        return "Erreur : la collection n'est pas initialisee."

    where_conditions = []
    if doc_name and str(doc_name).strip():
        where_conditions.append({"doc": {"$eq": str(doc_name).strip().lower()}})

    # On ne filtre pas par annee dans ChromaDB pour ne pas perdre les en-tetes de tableaux

    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}

    if _DEBUG_MODE:
        print(f"\n[DEBUG Outil] 'search_database' appele :")
        print(f"  -> requete : '{query}'")
        print(f"  -> document : '{doc_name}'")

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
            result_str = "Aucun document trouve correspondant aux criteres."
        else:
            ctx_block = []
            for i, (txt, meta) in enumerate(zip(docs, metas), 1):
                m_year = meta.get("year", "Inconnu")
                m_doc = meta.get("doc", "Inconnu")
                header = f"--- [Contexte {i} | Document : {m_doc} | Annee metadonnee : {m_year}] ---"
                ctx_block.append(f"{header}\n{txt}\n")
            result_str = "\n".join(ctx_block)

        return result_str

    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"


# Schema JSON de l'outil pour Groq/Gemini
DATABASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Recherche dans la base de donnees vectorielle des fragments de rapports financiers/ESG.",
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


# =========================
# Logique de l'agent
# =========================

def get_api_key() -> str:
    """Recupere la cle API Groq."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY non definie. Obtenez-en une sur console.groq.com"
        )
    return key


def run_agent(
    question: str,
    model_name: str,
    temperature: float,
    answer_style: str,
) -> tuple[str, int]:
    """Execute l'agent RAG avec fallback multi-modele (Groq/Gemini).

    Retourne (reponse, nombre_de_recherches).
    """
    groq_api_key = get_api_key()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Instructions selon le style de reponse
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
        "1. Appelez TOUJOURS search_database. Ne dites JAMAIS 'je n'ai pas acces' — vous AVEZ acces via l'outil.\n"
        "2. Ne demandez JAMAIS a l'utilisateur de chercher. VOUS devez continuer a chercher jusqu'a trouver.\n"
        "3. Ne repondez JAMAIS avec des phrases quand answer_style est 'value'. Repondez UNIQUEMENT avec le nombre.\n"
        "4. Si la premiere recherche ne donne rien d'utile, REESSAYEZ avec d'autres mots-cles ou sans filtre doc_name.\n"
        "5. Les tableaux utilisent '|' comme separateur. L'en-tete montre les annees (ex: 2021 | 2020 | 2019). "
        "Associez l'annee demandee a la bonne colonne SOIGNEUSEMENT.\n"
        "6. Les rapports peuvent contenir des donnees historiques pour plusieurs annees.\n"
        "7. Si vous voyez la metrique mais ne pouvez pas identifier la colonne, donnez votre meilleure estimation.\n"
        "8. Pour les sommes (ex: 'Total Scope 1 et 2'), cherchez chaque composante et calculez la somme.\n"
        f"9. {behavior_instr}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]

    # Prefixe par defaut
    if not model_name.startswith("groq:") and not model_name.startswith("gemini:"):
        model_name = f"groq:{model_name}"

    models_queue = [model_name] + [
        m
        for m in [
            "groq:llama-3.3-70b-versatile",
            "gemini:gemini-2.5-flash",
            "gemini:gemini-2.0-flash",
            "gemini:gemini-2.0-flash-lite-001",
            "groq:meta-llama/llama-4-scout-17b-16e-instruct",
            "groq:llama-3.1-8b-instant",
            "groq:qwen/qwen3-32b",
        ]
        if m != model_name
    ]

    # Boucle de l'agent
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
                        print(f"\n[DEBUG] {target_model} ignore (GEMINI_API_KEY non definie).")
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
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate" in err_str or "quota" in err_str

                if is_rate_limit and attempts_this_step < 2:
                    attempts_this_step += 1
                    if _DEBUG_MODE:
                        print(
                            f"\n[ATTENTION] Rate limit sur {target_model}, "
                            f"attente 15s (tentative {attempts_this_step}/2)..."
                        )
                    time.sleep(15)
                    continue

                failed_model = models_queue.pop(0)
                attempts_this_step = 0
                if _DEBUG_MODE:
                    print(
                        f"\n[ATTENTION] Modele {failed_model} en echec ({err_str[:80]}...). "
                        f"Bascule vers {models_queue[0] if models_queue else 'Aucun'}..."
                    )
                time.sleep(2)

        if not models_queue or response is None:
            return "ERREUR API : Rate limit depasse pour tous les modeles de secours.", num_searches

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content, num_searches

        # Serialiser en dict — fonctionne pour Groq et Gemini
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
                    year=args.get("year", 0),
                )
                num_searches += 1

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": tool_resp,
                    }
                )

    return "L'agent a epuise le nombre maximum d'etapes sans formuler de reponse.", num_searches


# =========================
# Point d'entree
# =========================

def main():
    global _DEBUG_MODE

    ap = argparse.ArgumentParser(description="Agent RAG Groq (TF-IDF + SVD)")
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT))

    ap.add_argument("--q", required=True, help="Question pour l'agent")
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
            print(f"\n[DEBUG] TOTAL RECHERCHES : {num_searches}")
            print(f"\n[DEBUG] REPONSE FINALE :\n{'-'*40}\n{answer}\n{'-'*40}\n")
        else:
            sys.stdout.write(answer + "\n")

    except Exception as e:
        sys.stderr.write(f"Erreur de l'agent : {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
