import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR = Path(os.environ.get("LOCALAPPDATA")) / "rag-activeviam" / "chroma"
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="data_ret_contexts_v2_chunks_meta", embedding_function=embedding_fn)

def search(q):
    res = collection.query(query_texts=[q], n_results=10)
    print(f"\nQUERY: {q}")
    for i, (doc, meta) in enumerate(zip(res['documents'][0], res['metadatas'][0])):
        print(f"[{meta['doc']} | {meta['year']}] {doc[:150].replace(chr(10), ' ')}")

search("Female directors")
search("Female directors 2021")
