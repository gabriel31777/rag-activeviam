"""
08_eval_agent.py
- Evaluates the accuracy of the Agent on the ActiveViam dataset.
- Picks RANDOM questions (with optional seed for reproducibility).
- Shows ALL questions with answers, marking ✅ or ❌.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import importlib

sys.path.append(str(Path(__file__).parent))

eval_v2 = importlib.import_module("04_eval_retrieval_v2")
value_matches = eval_v2.value_matches

agent_module = importlib.import_module("09_rag_agent_groq")
run_agent = agent_module.run_agent
init_collection = agent_module.init_collection
CHROMA_DIR_DEFAULT = agent_module.CHROMA_DIR_DEFAULT
COLLECTION_DEFAULT = agent_module.COLLECTION_DEFAULT
EMBEDDING_MODEL_DEFAULT = agent_module.EMBEDDING_MODEL_DEFAULT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"


def eval_agent(limit: int = 15, model: str = "llama-3.3-70b-versatile", seed: int = None):
    print(f"[INFO] Initializing RAG database...")
    init_collection(Path(CHROMA_DIR_DEFAULT), COLLECTION_DEFAULT, EMBEDDING_MODEL_DEFAULT)

    print(f"[INFO] Loading questions from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    # Drop questions whose gold answer is 0 — they add no signal for retrieval quality
    before = len(df)
    df = df[df["Value"].astype(str).str.strip().apply(
        lambda v: v not in ("0", "0.0", "0.00")
    )].reset_index(drop=True)
    print(f"[INFO] Filtered out {before - len(df)} zero-value questions ({len(df)} remaining).")

    # Random sampling (from the filtered pool)
    if seed is not None:
        print(f"[INFO] Using seed: {seed}")
    else:
        import random
        seed = random.randint(0, 99999)
        print(f"[INFO] Random seed (use --seed {seed} to reproduce): {seed}")

    df_eval = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)

    hits = 0
    total = len(df_eval)

    print(f"[INFO] Testing {total} samples using {model}...\n")

    for i, row in df_eval.iterrows():
        q = str(row["Question"])
        gold = str(row["Value"])

        try:
            agent_answer, n_searches = run_agent(q, model_name=model, temperature=0.1, answer_style="value")
        except Exception as e:
            agent_answer = f"API ERROR: {str(e)}"
            n_searches = 0

        ok = value_matches(gold, agent_answer)
        if ok:
            hits += 1

        status = "✅" if ok else "❌"
        print(f"  {status} [{i+1}/{total}] Q: {q}")
        print(f"       Gold: {gold}  |  Agent: {agent_answer}  (Searches: {n_searches})")
        print()

        time.sleep(30)

    rate = hits / total if total > 0 else 0
    print("=" * 60)
    print(f"  RESULTS — ActiveViam Dataset")
    print(f"  Model: {model}  |  Seed: {seed}")
    print(f"  Hits: {hits}/{total}  |  Accuracy: {rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    args = ap.parse_args()

    eval_agent(limit=args.limit, model=args.model, seed=args.seed)
