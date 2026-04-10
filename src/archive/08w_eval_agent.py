"""
08w_eval_agent.py (version TF-IDF + SVD)
Evalue la precision de l'agent RAG sur le dataset ActiveViam.

- Selectionne des questions ALEATOIRES (avec graine optionnelle pour reproductibilite)
- Affiche chaque question avec la reponse de l'agent et le resultat [OK] ou [ERREUR]
- Calcule le taux de precision global
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import importlib

sys.path.append(str(Path(__file__).parent))

# Import dynamique des modules necessaires
eval_v2 = importlib.import_module("04w_eval_retrieval_v2")
value_matches = eval_v2.value_matches

agent_module = importlib.import_module("09w_rag_agent_groq")
run_agent = agent_module.run_agent
init_collection = agent_module.init_collection
CHROMA_DIR_DEFAULT = agent_module.CHROMA_DIR_DEFAULT
COLLECTION_DEFAULT = agent_module.COLLECTION_DEFAULT
MODEL_PATH_DEFAULT = agent_module.MODEL_PATH_DEFAULT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"


def eval_agent(limit: int = 15, model: str = "llama-3.3-70b-versatile", seed: int = None):
    """Evalue l'agent sur un echantillon aleatoire du dataset."""
    print("[INFO] Initialisation de la base RAG...")
    init_collection(Path(CHROMA_DIR_DEFAULT), COLLECTION_DEFAULT, Path(MODEL_PATH_DEFAULT))

    print(f"[INFO] Chargement des questions depuis {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    # Filtrer les questions dont la valeur attendue est 0 (pas de signal utile)
    before = len(df)
    df = df[
        df["Value"]
        .astype(str)
        .str.strip()
        .apply(lambda v: v not in ("0", "0.0", "0.00"))
    ].reset_index(drop=True)
    print(f"[INFO] {before - len(df)} questions a valeur zero filtrees ({len(df)} restantes).")

    # Echantillonnage aleatoire
    if seed is not None:
        print(f"[INFO] Graine utilisee : {seed}")
    else:
        import random
        seed = random.randint(0, 99999)
        print(f"[INFO] Graine aleatoire (pour reproduire : --seed {seed}) : {seed}")

    df_eval = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)

    hits = 0
    total = len(df_eval)

    print(f"[INFO] Test de {total} echantillons avec le modele {model}...\n")

    for i, row in df_eval.iterrows():
        q = str(row["Question"])
        gold = str(row["Value"])

        try:
            agent_answer, n_searches = run_agent(q, model_name=model, temperature=0.1, answer_style="value")
        except Exception as e:
            agent_answer = f"ERREUR API : {str(e)}"
            n_searches = 0

        ok = value_matches(gold, agent_answer)
        if ok:
            hits += 1

        status = "[OK]" if ok else "[ERREUR]"
        print(f"  {status} [{i+1}/{total}] Q : {q}")
        print(f"       Attendu : {gold}  |  Agent : {agent_answer}  (Recherches : {n_searches})")
        print()

        # Pause pour eviter le rate limiting
        time.sleep(30)

    rate = hits / total if total > 0 else 0
    print("=" * 60)
    print(f"  RESULTATS -- Dataset ActiveViam")
    print(f"  Modele : {model}  |  Graine : {seed}")
    print(f"  Reussis : {hits}/{total}  |  Precision : {rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluation de l'agent RAG")
    ap.add_argument("--limit", type=int, default=15, help="Nombre de questions a tester")
    ap.add_argument("--model", default="llama-3.3-70b-versatile", help="Modele LLM a utiliser")
    ap.add_argument("--seed", type=int, default=None, help="Graine pour reproductibilite")
    args = ap.parse_args()

    eval_agent(limit=args.limit, model=args.model, seed=args.seed)
