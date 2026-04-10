"""
11_eval_finqa.py
- Evaluates the Agent on the FinQA dataset.
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
import os

sys.path.append(str(Path(__file__).parent))

eval_v2 = importlib.import_module("04_eval_retrieval_v2")
value_matches = eval_v2.value_matches

agent_module = importlib.import_module("09_rag_agent_groq")
run_agent = agent_module.run_agent
init_collection = agent_module.init_collection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "finqa_clean.csv"

LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR = (
    Path(LOCALAPPDATA) / "rag-activeviam" / "chroma" if LOCALAPPDATA else (PROJECT_ROOT / "data" / "chroma")
)
COLLECTION_NAME = "finqa_chunks_meta"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def eval_finqa(limit: int = 15, model: str = "llama-3.1-8b-instant", split: str = "dev", seed: int = None):
    print(f"[INFO] Initializing FinQA RAG database...")
    init_collection(Path(CHROMA_DIR), COLLECTION_NAME, EMBEDDING_MODEL)

    print(f"[INFO] Loading questions from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    if "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
        print(f"[INFO] Filtered to '{split}' split: {len(df)} rows")

    # Random sampling
    if seed is not None:
        print(f"[INFO] Using seed: {seed}")
    else:
        import random
        seed = random.randint(0, 99999)
        print(f"[INFO] Random seed (use --seed {seed} to reproduce): {seed}")

    df_eval = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)

    hits = 0
    total = len(df_eval)

    print(f"[INFO] Testing {total} FinQA samples using {model}...\n")

    for i, row in df_eval.iterrows():
        q = str(row["Question"])
        gold = str(row["Value"])

        try:
            agent_answer = run_agent(q, model_name=model, temperature=0.1, answer_style="value")
        except Exception as e:
            agent_answer = f"API ERROR: {str(e)}"

        ok = value_matches(gold, agent_answer)
        if ok:
            hits += 1

        status = "✅" if ok else "❌"
        print(f"  {status} [{i+1}/{total}] Q: {q}")
        print(f"       Gold: {gold}  |  Agent: {agent_answer}")
        print()

        time.sleep(2)

    rate = hits / total if total > 0 else 0
    print("=" * 60)
    print(f"  RESULTS — FinQA Dataset ({split} split)")
    print(f"  Model: {model}  |  Seed: {seed}")
    print(f"  Hits: {hits}/{total}  |  Accuracy: {rate * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    args = ap.parse_args()

    eval_finqa(limit=args.limit, model=args.model, split=args.split, seed=args.seed)
