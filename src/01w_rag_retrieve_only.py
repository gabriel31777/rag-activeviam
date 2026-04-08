"""
01w_rag_retrieve_only.py - Word2Vec Version

Exact replica of 01_rag_retrieve_only.py but using Word2Vec embeddings instead of sentence-transformers.
- Reads the dataset (CSV cleaned) with columns: Question, Context, Value, prompt
- Creates embeddings of Context (1 doc per line, for now)
- Indexes in ChromaDB (persistent at ./data/chroma)
- Performs a retrieve search for a test question
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

import chromadb
import pickle

# =========================
# Config (paths and parameters)
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "Data_ret.csv"
MODEL_PATH = Path.home() / "AppData" / "Local" / "rag-activeviam" / "models" / "word2vec_pdf.pkl"
CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "chroma_w2v"
COLLECTION_NAME = "data_ret_contexts_v1_w2v"

TOP_K = 5
ADD_BATCH_SIZE = 500


# =========================
# Utility Functions
# =========================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and clean the CSV dataset."""
    df = pd.read_csv(csv_path)
    
    # Drop unnamed columns
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])
    
    # Require these columns
    required_cols = {"Question", "Context", "Value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    print(f"[INFO] Loaded {len(df)} rows from {csv_path.name}")
    return df


def load_word2vec_model(model_path: Path):
    """Load the Word2Vec model (TF-IDF + SVD)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"[INFO] Loaded Word2Vec model from {model_path}")
    return model_data


class Word2VecEmbeddingFunction:
    """ChromaDB-compatible embedding function for Word2Vec."""
    
    def __init__(self, model_data: dict):
        self.vectorizer = model_data['#Data_ret   self.reducer = model_data['reducer']  # SVD
        self.vectors = model_data['vectors']  # Pre-computed document vectors
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed texts using the Word2Vec model."""
        if not input:
            return []
        
        # Transform texts using TF-IDF and SVD
        tfidf_vecs = self.vectorizer.transform(input)
        embeddings = self.reducer.transform(tfidf_vecs).tolist()
        return embeddings
    
    def embed_query(self, input: str) -> List[float]:
        """Embed a single query."""
        return self([input])[0]
    
    @property
    def name(self) -> str:
        return "word2vec_tfidf_svd"


def make_documents(df: pd.DataFrame) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Create documents, IDs, and metadata from DataFrame."""
    doc_texts = []
    doc_ids = []
    doc_metadatas = []
    
    for idx, row in df.iterrows():
        doc_texts.append(str(row["Context"]))
        doc_ids.append(f"doc_{idx}")
        
        metadata = {
            "row_index": int(idx),
            "question": str(row.get("Question", "")),
            "value": str(row.get("Value", "")),
        }
        doc_metadatas.append(metadata)
    
    return doc_texts, doc_ids, doc_metadatas


def build_or_load_collection(
    chroma_dir: Path,
    collection_name: str,
    embedding_func: Word2VecEmbeddingFunction,
) -> chromadb.Collection:
    """Create or load a ChromaDB collection."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Try to load existing collection
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        print(f"[INFO] Loaded existing collection '{collection_name}'")
        return collection
    except:
        print(f"[INFO] Creating new collection '{collection_name}'")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        return collection


def add_to_collection_in_batches(
    collection: chromadb.Collection,
    documents: List[str],
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 500,
) -> None:
    """Add documents to collection in batches."""
    total = len(documents)
    
    for i in tqdm(range(0, total, batch_size), desc="Indexing documents"):
        end = min(i + batch_size, total)
        collection.add(
            documents=documents[i:end],
            ids=ids[i:end],
            metadatas=metadatas[i:end]
        )
    
    print(f"[INFO] Successfully indexed {total} documents")


def retrieve(
    collection: chromadb.Collection,
    query: str,
    n_results: int = TOP_K,
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """Retrieve documents for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    
    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    return documents, distances, metadatas


# =========================
# Main
# =========================

def main():
    print("\n" + "=" * 70)
    print("01w_rag_retrieve_only.py - Word2Vec Version")
    print("=" * 70)
    
    # Load dataset
    df = load_dataset(CSV_PATH)
    
    # Load Word2Vec model
    model_data = load_word2vec_model(MODEL_PATH)
    embedding_func = Word2VecEmbeddingFunction(model_data)
    
    # Create documents
    print(f"\n[INFO] Creating {len(df)} documents...")
    doc_texts, doc_ids, doc_metadatas = make_documents(df)
    
    # Build or load collection
    print(f"\n[INFO] Initializing ChromaDB collection...")
    collection = build_or_load_collection(
        CHROMA_DIR,
        COLLECTION_NAME,
        embedding_func
    )
    
    # Check if collection is empty
    if collection.count() == 0:
        print(f"\n[INFO] Collection is empty, adding documents...")
        add_to_collection_in_batches(
            collection,
            doc_texts,
            doc_ids,
            doc_metadatas,
            batch_size=ADD_BATCH_SIZE
        )
    else:
        print(f"[INFO] Collection already has {collection.count()} documents")
    
    # Test retrieve
    print(f"\n[INFO] Testing retrieval...")
    test_query = "What is ESG?"
    documents, distances, metadatas = retrieve(collection, test_query, n_results=5)
    
    print(f"\nQuery: {test_query}")
    print(f"Results (Top {len(documents)}):")
    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
        similarity = 1 - dist
        print(f"\n  [{i+1}] Similarity: {similarity:.4f}")
        print(f"      Question: {meta.get('question', 'N/A')[:60]}...")
        print(f"      Context: {doc[:100]}...")
        print(f"      Value: {meta.get('value', 'N/A')}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
