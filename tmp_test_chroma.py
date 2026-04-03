import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parents[0]
LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
CHROMA_DIR = Path(LOCALAPPDATA) / "rag-activeviam" / "chroma" if LOCALAPPDATA else (PROJECT_ROOT / "data" / "chroma")
COLLECTION_NAME = "data_ret_contexts_v2_chunks_meta"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading Chroma from {CHROMA_DIR}...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)
print(f"Collection count: {collection.count()}")

query = "Number of person hours worked Tongaat 2021"
print(f"Querying: {query}...")
res = collection.query(query_texts=[query], n_results=5)
print(f"Found {len(res['documents'][0])} results.")
for i, doc in enumerate(res['documents'][0]):
    print(f"Result {i+1}: {doc[:100]}...")
