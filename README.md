# RAG ActiveViam

Small Streamlit application for exploring Retrieval-Augmented Generation (RAG) on private document collections.

The app lets you:

- create datasets of uploaded documents
- build a vector index for retrieval
- chat with the indexed content
- inspect extracted document text

It is designed as a simple RAG playground with a few retrieval strategies and a lightweight UI.

## Main Features

- Streamlit interface for chatting with document collections
- dataset management page for creating, uploading, deleting, and indexing data
- hybrid retrieval based on dense vectors + BM25 + reranking
- alternative retrieval modes for table-of-contents navigation and raw PDF page retrieval
- source citations and prompt transparency in the chat UI
- persistent ChromaDB storage for document embeddings

## Project Structure

```text
.
├── app/
│   ├── app.py                  # Streamlit entrypoint and page navigation
│   ├── core/                   # Retrieval, indexing, chunking, embeddings
│   ├── dataset/                # Dataset storage and document loading
│   ├── llm/                    # LLM abstraction and Gemini implementation
│   ├── pages/                  # Streamlit pages
│   ├── services/               # App-level orchestration used by pages
│   └── utils/                  # Config and logging helpers
├── docker/
│   └── Dockerfile              # Container image for the app
├── docker-compose.yml          # Local container orchestration
└── requirements.txt            # Python dependencies
```

## Architecture Overview

The codebase is split into a few clear layers:

- `pages/`: Streamlit UI only. These files render widgets and display results.
- `services/`: application use cases. They connect the UI to the lower-level modules.
- `core/`: RAG internals such as chunking, retrieval, embeddings, vector storage, and pipeline orchestration.
- `dataset/`: filesystem-based dataset management and document parsing.
- `llm/`: language model interface and Gemini integration.
- `utils/`: configuration and logging.

This keeps UI concerns separate from indexing and retrieval logic.

## Core Flows

### 1. Dataset Ingestion

When a user builds an index for a dataset:

1. documents are loaded from disk
2. documents are chunked
3. chunk embeddings are generated
4. chunks and embeddings are stored in ChromaDB
5. index metadata is saved for later compatibility checks

The main orchestration lives in `app/core/rag_pipeline.py`.

### 2. Question Answering

When a user asks a question:

1. a retrieval mode is selected
2. relevant sources are fetched from the dataset
3. a prompt is built from the retrieved context
4. the LLM generates an answer
5. the UI shows the answer, sources, and optionally the full prompt

## Retrieval Modes

The app currently supports three retrieval modes:

- `vector`: the default hybrid flow using embeddings, BM25, reciprocal rank fusion, and cross-encoder reranking
- `page_index`: retrieval based on document structure and headings
- `pdf_raw`: retrieval of raw PDF pages reconstructed from indexed content

## Important Modules

- `app/core/rag_pipeline.py`: main ingestion and query orchestration
- `app/core/retriever.py`: hybrid retriever with reranking
- `app/core/vector_store.py`: ChromaDB wrapper
- `app/dataset/manager.py`: dataset CRUD and file persistence
- `app/dataset/loaders.py`: PDF, DOCX, TXT, Markdown, and HTML loading
- `app/services/`: thin orchestration layer used by the Streamlit pages

## Running Locally

### Requirements

- Python 3.11 recommended
- a valid `GEMINI_API_KEY`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with at least:

```env
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini
```

Optional paths:

```env
DOCUMENTS_PATH=./data/documents
VECTORDB_PATH=./data/vectordb
```

### Start the App

```bash
cd app
streamlit run app.py
```

## Running with Docker

Build and start:

```bash
docker compose up --build
```

The app is exposed on `http://localhost:8501`.

Document storage and vector storage are persisted through Docker volumes.

## Dependencies

Main libraries used in the project:

- `streamlit` for the UI
- `chromadb` for vector persistence
- `sentence-transformers` for embeddings and reranking
- `google-genai` for the LLM
- `pymupdf` and `pymupdf4llm` for PDF extraction
- `python-docx` for DOCX extraction
- `rank_bm25` for sparse retrieval

## Notes for Contributors

- `services/` is the best place for page-level business logic
- `pages/` should stay focused on rendering and Streamlit state
- `core/` should stay framework-agnostic when possible
- `dataset/` owns filesystem and file parsing concerns

If you want to extend the project, a good next step would be adding tests around the `services/` and `core/` layers.
