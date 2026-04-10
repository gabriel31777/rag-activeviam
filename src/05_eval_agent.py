"""
05_eval_agent.py
Évalue la précision de l'agent RAG end-to-end.

Le CSV fournit les questions et valeurs attendues (gabarito).
L'agent cherche dans les PDFs indexés dans ChromaDB.

Utilisation :
    python src/05_eval_agent.py --embedding tfidf_svd --limit 20
    python src/05_eval_agent.py --embedding word2vec --limit 20
    python src/05_eval_agent.py --embedding sentence_transformer --limit 10
    python src/05_eval_agent.py --embedding tfidf_svd --limit 20 --seed 42
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importation dynamique des modules
agent_module = importlib.import_module("04_rag_agent")
run_agent = agent_module.run_agent
init_collection = agent_module.init_collection

# Importation de value_matches depuis le module d'évaluation du retrieval
eval_module = importlib.import_module("03_eval_retrieval")
value_matches = eval_module.value_matches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"


def eval_agent(
    limit: int = 15,
    model: str = "llama-3.3-70b-versatile",
    seed: int = None,
    embedding: str = "tfidf_svd",
):
    """Évalue l'agent sur un échantillon aléatoire du dataset."""

    print(f"[INFO] Initialisation ({embedding})...")
    init_collection(embedding)

    print(f"[INFO] Chargement du gabarito : {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    # Filtrer les questions à valeur zéro
    before = len(df)
    df = df[
        df["Value"]
        .astype(str)
        .str.strip()
        .apply(lambda v: v not in ("0", "0.0", "0.00"))
    ].reset_index(drop=True)
    print(f"[INFO] {before - len(df)} questions à valeur zéro filtrées ({len(df)} restantes).")

    # Échantillonnage aléatoire
    if seed is not None:
        print(f"[INFO] Graine : {seed}")
    else:
        import random
        seed = random.randint(0, 99999)
        print(f"[INFO] Graine aléatoire (pour reproduire : --seed {seed})")

    df_eval = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)

    hits = 0
    total = len(df_eval)

    emb_label = {
        "tfidf_svd": "TF-IDF + SVD",
        "word2vec": "Word2Vec (gensim)",
        "sentence_transformer": "SentenceTransformers",
    }.get(embedding, embedding)

    print(f"[INFO] Test de {total} questions avec {model} + {emb_label}\n")

    for i, row in df_eval.iterrows():
        q = str(row["Question"])
        gold = str(row["Value"])

        try:
            agent_answer, n_searches = run_agent(
                q, model_name=model, temperature=0.1, answer_style="value"
            )
        except Exception as e:
            agent_answer = f"ERREUR : {str(e)}"
            n_searches = 0

        ok = value_matches(gold, agent_answer)
        if ok:
            hits += 1

        status = "[OK]" if ok else "[ERREUR]"
        print(f"  {status} [{i+1}/{total}] Q : {q}")
        print(f"       Attendu : {gold}  |  Agent : {agent_answer}  (Recherches : {n_searches})")
        print()

        # Pause pour éviter le rate limiting
        time.sleep(30)

    rate = hits / total if total > 0 else 0
    print("=" * 60)
    print(f"  RÉSULTATS — {emb_label}")
    print(f"  Modèle : {model}  |  Graine : {seed}")
    print(f"  Réussis : {hits}/{total}  |  Précision : {rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Évaluation de l'agent RAG")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--embedding",
        choices=["tfidf_svd", "word2vec", "sentence_transformer", "hybrid"],
        default="tfidf_svd",
    )
    args = ap.parse_args()

    eval_agent(limit=args.limit, model=args.model, seed=args.seed, embedding=args.embedding)
