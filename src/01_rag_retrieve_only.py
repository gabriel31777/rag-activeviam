"""
01_rag_retrieve_only.py
- Lê o dataset (CSV limpo) com colunas: Question, Context, Value, prompt
- Cria embeddings do Context (1 doc por linha, por enquanto)
- Indexa no ChromaDB (persistente em ./data/chroma)
- Faz uma busca (retrieve) para uma pergunta de teste
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# =========================
# Config (paths e parâmetros)
# =========================

# Pasta raiz do projeto = pasta onde está o script (src) -> pai
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ajuste aqui se quiser outro arquivo:
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# Onde o Chroma vai persistir o índice
CHROMA_DIR = Path(os.environ["LOCALAPPDATA"]) / "rag-activiam" / "chroma"

# Nome da coleção no Chroma
COLLECTION_NAME = "data_ret_contexts_v1"

# Modelo de embeddings (leve, funciona bem para começo)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Quantos resultados trazer por busca
TOP_K = 5

# Tamanho do batch de inserção no Chroma
# (mantemos bem abaixo do limite para evitar erro)
ADD_BATCH_SIZE = 500


# =========================
# Funções utilitárias
# =========================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Não achei o CSV em: {csv_path}\n"
            f"Verifique se o arquivo existe e se o caminho está correto."
        )

    df = pd.read_csv(csv_path)

    # Limpeza básica: remover colunas tipo "Unnamed: 0" se existirem
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])

    # Garantir colunas esperadas
    required = {"Question", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV não tem colunas obrigatórias: {missing}. Colunas atuais: {list(df.columns)}")

    # Remover linhas vazias no Context (por segurança)
    df["Context"] = df["Context"].astype(str)
    df = df[df["Context"].str.strip().ne("")].reset_index(drop=True)

    return df


def make_documents(df: pd.DataFrame) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    1 documento por linha, usando o campo Context.
    ids: ctx_000001, ctx_000002, ...
    metadados: inclui a pergunta original (para debug) e índice
    """
    documents: List[str] = df["Context"].astype(str).tolist()
    ids: List[str] = [f"ctx_{i:06d}" for i in range(len(documents))]

    metadatas: List[Dict[str, Any]] = []
    if "Question" in df.columns:
        questions = df["Question"].astype(str).tolist()
    else:
        questions = [""] * len(documents)

    for i in range(len(documents)):
        metadatas.append({
            "row_index": int(i),
            "question": questions[i],
        })

    return documents, ids, metadatas


def batched(iterable: List[Any], batch_size: int):
    """Gera fatias (slices) de uma lista em batches."""
    for start in range(0, len(iterable), batch_size):
        end = start + batch_size
        yield start, end, iterable[start:end]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_or_load_collection() -> Any:
    """
    Cria cliente persistente e coleção.
    Se já existir, reusa.
    """
    ensure_dir(CHROMA_DIR)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # get_or_create_collection evita erro se já existir
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def maybe_reset_collection(collection) -> None:
    """
    Se você quiser sempre reconstruir do zero, descomente o conteúdo abaixo.
    Por padrão, NÃO apagamos nada.
    """
    # client = collection._client  # não recomendado mexer internals
    # client.delete_collection(COLLECTION_NAME)
    # print("Coleção apagada.")


def add_to_collection_in_batches(collection, documents: List[str], ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
    """
    Adiciona docs em batches menores para evitar limite interno do Chroma.
    Também checa se já tem dados (para não duplicar).
    """
    current_count = collection.count()
    if current_count > 0:
        print(f"[INFO] Coleção já tem {current_count} itens. Vou pular indexação para evitar duplicatas.")
        print("       Se quiser reindexar do zero, apague a pasta data/chroma ou mude COLLECTION_NAME.")
        return

    print(f"[INFO] Indexando {len(documents)} documentos em batches de {ADD_BATCH_SIZE}...")

    # Vamos iterar por índice para cortar docs/ids/metadatas alinhados
    for start in tqdm(range(0, len(documents), ADD_BATCH_SIZE)):
        end = min(start + ADD_BATCH_SIZE, len(documents))

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas,
        )

    print("[INFO] Indexação concluída.")


def retrieve(collection, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    # results é um dict com listas (1 query -> índice 0)
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    ids_ = results["ids"][0]

    out: List[Dict[str, Any]] = []
    for doc, dist, meta, _id in zip(docs, dists, metas, ids_):
        out.append({
            "id": _id,
            "distance": float(dist),
            "metadata": meta,
            "document_preview": (doc[:300] + "..." if len(doc) > 300 else doc),
        })
    return out


# =========================
# Main
# =========================

def main():
    print("[INFO] Projeto:", PROJECT_ROOT)
    print("[INFO] CSV:", CSV_PATH)
    print("[INFO] Chroma dir:", CHROMA_DIR)

    df = load_dataset(CSV_PATH)
    print(f"[INFO] Dataset carregado: {len(df)} linhas | colunas: {list(df.columns)}")

    documents, ids, metadatas = make_documents(df)

    collection = build_or_load_collection()

    add_to_collection_in_batches(collection, documents, ids, metadatas)

    print(f"[INFO] Total na coleção agora: {collection.count()}")

    # Pergunta de teste (você pode trocar)
    test_query = "What is the main financial value mentioned?"
    print("\n[TEST] Query:", test_query)

    hits = retrieve(collection, test_query, top_k=TOP_K)
    for i, h in enumerate(hits, 1):
        print(f"\n--- Hit {i} ---")
        print("ID:", h["id"])
        print("Distance:", h["distance"])
        print("Metadata:", h["metadata"])
        print("Doc preview:", h["document_preview"])


if __name__ == "__main__":
    main()
