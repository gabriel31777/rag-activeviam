"""
06w_rag_generate_gemini.py (version TF-IDF + SVD)
Generation RAG via l'API Gemini (Google GenAI).

Pipeline :
  1. Charge la collection ChromaDB (embedding TF-IDF + SVD)
  2. Effectue le retrieval pour la question posee
  3. Construit un prompt avec les contextes recuperes
  4. Envoie le prompt a Gemini pour generer la reponse
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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

CSV_PATH_DEFAULT = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"
CHROMA_DIR_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v"
COLLECTION_DEFAULT = "data_ret_contexts_v2_chunks_meta_w2v"
MODEL_PATH_DEFAULT = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models" / "word2vec_pdf.pkl"

# Expressions regulieres pour extraire le document/annee et la metrique
DOC_YEAR_RE = re.compile(r"in\s+(.+?)\s+document\s+in\s+(\d{4})\??$", re.IGNORECASE)
VALUE_OF_RE = re.compile(r"^(?:the\s+)?value\s+of\s+(.+?)\s+in\s+", re.IGNORECASE)


# =========================
# Fonctions utilitaires
# =========================

def parse_doc_year(question: str) -> Tuple[Optional[str], Optional[int]]:
    """Extrait le nom du document et l'annee depuis la question."""
    q = (question or "").strip()
    m = DOC_YEAR_RE.search(q)
    if not m:
        return None, None
    doc = m.group(1).strip().lower()
    year = int(m.group(2))
    return doc, year


def extract_metric(question: str) -> Optional[str]:
    """Extrait le nom de la metrique depuis la question."""
    q = (question or "").strip()
    m = VALUE_OF_RE.match(q)
    if not m:
        return None
    return m.group(1).strip()


# =========================
# ChromaDB
# =========================

def build_collection(chroma_dir: Path, collection_name: str, model_path: Path) -> Any:
    """Charge la collection ChromaDB avec l'embedding TF-IDF + SVD."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = TfidfSvdEmbeddingFunction(model_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_where_filter(doc: Optional[str], year: Optional[int]) -> Optional[Dict[str, Any]]:
    """Construit le filtre ChromaDB."""
    if doc is None or year is None:
        return None
    return {"$and": [{"doc": {"$eq": doc}}, {"year": {"$eq": int(year)}}]}


def retrieve(
    collection: Any,
    question: str,
    top_k: int,
    where_filter: Optional[Dict[str, Any]],
    retrieval_query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Effectue le retrieval — utilise la metrique comme texte de requete si disponible."""
    qt = retrieval_query_text.strip() if retrieval_query_text else question

    kwargs = dict(
        query_texts=[qt],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )
    if where_filter is not None:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


# =========================
# API Gemini
# =========================

def get_api_key() -> str:
    """Recupere la cle API Gemini depuis les variables d'environnement."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY (ou GOOGLE_API_KEY) non definie dans l'environnement."
        )
    return key


def call_gemini(
    prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Tuple[str, Any, Any]:
    """Appelle l'API Gemini et retourne la reponse textuelle."""
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
# Construction du prompt
# =========================

def build_prompt(
    question: str,
    contexts: List[str],
    metadatas: List[Dict],
    answer_style: str,
) -> str:
    """Construit le prompt final avec les contextes et les instructions."""
    target_doc, target_year = parse_doc_year(question)
    year_instruction = ""
    if target_year:
        year_instruction = f"L'ANNEE CIBLE EST {target_year}. CONCENTREZ-VOUS SUR LES COLONNES OU DONNEES DE {target_year}.\n"

    base_rules = (
        "Vous etes un analyste de donnees intelligent. Vous lisez des fragments de rapports financiers/ESG.\n"
        "Votre objectif est d'extraire une valeur specifique en reponse a la question.\n\n"
        "REGLES IMPORTANTES POUR LES TABLEAUX FRAGMENTES :\n"
        "1. Le texte est decoupe en chunks. Les en-tetes (ex: '2022 | 2021 | 2020') peuvent etre dans un chunk, et les donnees dans un autre.\n"
        "2. Cherchez les motifs. Si un en-tete definit l'ordre des colonnes, APPLIQUEZ cet ordre aux donnees des autres contextes.\n"
        "3. Les ordres de colonnes sont generalement decroissants (annee courante, puis precedente).\n"
        f"{year_instruction}"
        "4. Si la question demande une somme (ex: 'Total Scope 1 et 2'), et que vous voyez les composantes, faites la somme vous-meme.\n"
    )

    if answer_style == "value":
        style_instr = (
            "Format de sortie : Retournez UNIQUEMENT la valeur numerique.\n"
            "N'ecrivez pas de phrases. Ne dites pas 'je ne sais pas' sauf si la donnee est completement absente.\n"
            "Si vous etes sur a 70% grace a la structure du tableau, donnez la valeur.\n"
        )
    elif answer_style == "free":
        style_instr = (
            "Format de sortie : Repondez naturellement. Expliquez COMMENT vous avez trouve la valeur.\n"
        )
    else:
        style_instr = "Repondez naturellement."

    ctx_block = []
    for i, (txt, meta) in enumerate(zip(contexts, metadatas), 1):
        doc_year = meta.get("year", "Inconnu")
        doc_name = meta.get("doc", "Inconnu")
        header = f"--- [Contexte {i} | Document : {doc_name} | Annee : {doc_year}] ---"
        ctx_block.append(f"{header}\n{txt}\n")

    return (
        base_rules
        + "\n" + style_instr
        + "\n=== CONTEXTES ===\n"
        + "\n".join(ctx_block)
        + "\n=== FIN CONTEXTES ===\n\n"
        + "Question : " + question
        + "\nReponse :"
    )


# =========================
# Point d'entree
# =========================

def main():
    ap = argparse.ArgumentParser(description="Generation RAG via Gemini")
    ap.add_argument("--csv", default=str(CSV_PATH_DEFAULT))
    ap.add_argument("--chroma-dir", default=str(CHROMA_DIR_DEFAULT))
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT))

    ap.add_argument("--q", default=None, help="Question libre (texte)")
    ap.add_argument("--i", type=int, default=None, help="Index du CSV a utiliser")
    ap.add_argument("--auto-find", type=int, default=None, help="Selection automatique d'un index")

    ap.add_argument("--k", type=int, default=15, help="Nombre de contextes recuperes")
    ap.add_argument("--mode", choices=["chat", "debug"], default="chat")
    ap.add_argument("--answer-style", choices=["value", "short", "free"], default="value")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--max-out", type=int, default=8192, help="Tokens max en sortie")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    chroma_dir = Path(args.chroma_dir)
    model_path = Path(args.model_path)
    collection = build_collection(chroma_dir, args.collection, model_path)
    df = pd.read_csv(csv_path)

    # Determiner la question
    if args.q:
        question = args.q.strip()
    elif args.i is not None:
        question = str(df.loc[int(args.i), "Question"])
    elif args.auto_find is not None:
        question = str(df.loc[int(args.auto_find), "Question"])
    else:
        print("[ERREUR] Utilisez --q, --i ou --auto-find pour specifier une question.")
        return

    doc, year = parse_doc_year(question)
    where_filter = build_where_filter(doc, year)
    metric = extract_metric(question)

    # Retrieval
    res = retrieve(collection, question, args.k, where_filter, metric)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # Prompt et appel Gemini
    prompt = build_prompt(question, docs, metas, args.answer_style)
    answer, finish, usage = call_gemini(prompt, args.model, args.temp, args.max_out)

    # Affichage
    if args.mode == "chat":
        print(answer)
    else:
        print(f"[DEBUG] Question : {question}")
        print(f"[DEBUG] Filtre : {where_filter}")
        print(f"[DEBUG] Tokens (Entree/Sortie) : {usage.prompt_token_count}/{usage.candidates_token_count}")
        print(f"\n[DEBUG] PROMPT ENVOYE :\n{'-'*40}\n{prompt}\n{'-'*40}\n")
        print(f"[DEBUG] REPONSE GEMINI :\n{answer}")


if __name__ == "__main__":
    main()
