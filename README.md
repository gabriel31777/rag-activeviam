# RAG Activeviam -- Extraction de Donnees Financieres par IA

Systeme RAG (Retrieval-Augmented Generation) pour extraire des donnees financieres a partir de rapports PDF bruts. Utilise **ChromaDB** pour le stockage vectoriel, **PyMuPDF** pour l'extraction de texte/tableaux, et un agent LLM (Groq/Gemini) avec fallback automatique.

## Prerequis

- **Python 3.10+**
- **API Keys** :
  - [Groq](https://console.groq.com/keys) (primaire)
  - [Google Gemini](https://aistudio.google.com/app/apikey) (fallback)

## Installation

```bash
# 1. Creer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Mac/Linux

# 2. Configurer les cles API
cp .env.example .env           # puis remplir les cles

# 3. Installer les dependances
pip install -r requirements.txt
```

## Embeddings Disponibles

| Methode              | Flag                    | Description                         |
|----------------------|-------------------------|-------------------------------------|
| TF-IDF + SVD         | `--embedding tfidf_svd` | Rapide, algebrique                  |
| Word2Vec (gensim)    | `--embedding word2vec`  | Reseau de neurones sur le corpus    |
| SentenceTransformers | `--embedding sentence_transformer` | Modele pre-entraine, haute qualite |
| Hybrid (RRF)         | `--embedding hybrid`    | Combine TF-IDF + SentenceTransformers |

## Pipeline d'Execution

### Etape 1 : Entrainer les modeles locaux (TF-IDF+SVD et Word2Vec)
```bash
python src/01_train_embeddings.py
```
> Entraine les deux modeles a partir des PDFs dans `data/raw/Structured data/`.
> SentenceTransformers n'a pas besoin d'entrainement (modele pre-entraine).

### Etape 2 : Indexer les PDFs dans ChromaDB
```bash
python src/02_index_pdfs.py --embedding tfidf_svd --force
python src/02_index_pdfs.py --embedding word2vec --force
python src/02_index_pdfs.py --embedding sentence_transformer --force
```

### Etape 3 : Evaluer le retrieval (Hit@K, sans API)
```bash
python src/03_eval_retrieval.py --embedding tfidf_svd
python src/03_eval_retrieval.py --embedding word2vec
python src/03_eval_retrieval.py --embedding sentence_transformer
python src/03_eval_retrieval.py --embedding hybrid
```

### Etape 4 : Evaluer l'agent end-to-end (utilise API Groq/Gemini)
```bash
python src/05_eval_agent.py --embedding tfidf_svd --limit 20
python src/05_eval_agent.py --embedding word2vec --limit 20
python src/05_eval_agent.py --embedding sentence_transformer --limit 20
python src/05_eval_agent.py --embedding hybrid --limit 20
```

### Interface Web
```bash
python app.py
# Ouvrir http://127.0.0.1:5000
```

## Structure des Scripts

```
src/
├── embeddings/                     # Module d'embedding
│   ├── tfidf_svd_embedding.py      # TF-IDF + SVD
│   ├── word2vec_embedding.py       # Word2Vec (gensim)
│   ├── sentence_transformer_embedding.py  # SentenceTransformers
│   └── embedding_factory.py        # Factory
├── 01_train_embeddings.py          # Entraine TF-IDF+SVD et Word2Vec
├── 02_index_pdfs.py                # Indexe les PDFs dans ChromaDB
├── 03_eval_retrieval.py            # Evalue le retrieval (Hit@K)
├── 04_rag_agent.py                 # Agent RAG (Groq/Gemini)
└── 05_eval_agent.py                # Evaluation end-to-end
app.py                              # Interface web Flask
```

## Donnees

- `data/raw/Structured data/` : PDFs des rapports (source de donnees)
- `data/processed/data_ret_clean.csv` : Verite terrain (questions + reponses attendues, pour evaluation uniquement)
- `.env` : Cles API (ignore par Git)