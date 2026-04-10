# RAG Activeviam — Extraction de Données Financières par IA

Système RAG (Retrieval-Augmented Generation) pour extraire des données financières à partir de rapports PDF bruts. Utilise **ChromaDB** pour le stockage vectoriel, **PyMuPDF** pour l'extraction de texte/tableaux, et un agent LLM (Groq/Gemini) avec fallback automatique.

## Prérequis

- **Python 3.10+**
- **API Keys** :
  - [Groq](https://console.groq.com/keys) (primaire)
  - [Google Gemini](https://aistudio.google.com/app/apikey) (fallback)

## Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Mac/Linux

# 2. Configurer les clés API
cp .env.example .env           # puis remplir les clés

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Embeddings Disponibles

| Méthode              | Flag                    | Description                         |
|----------------------|-------------------------|-------------------------------------|
| TF-IDF + SVD         | `--embedding tfidf_svd` | Rapide, algébrique                  |
| Word2Vec (gensim)    | `--embedding word2vec`  | Réseau de neurones sur le corpus    |
| SentenceTransformers | `--embedding sentence_transformer` | Modèle pré-entraîné, haute qualité |

## Pipeline d'Exécution

### Étape 1 : Entraîner les modèles locaux (TF-IDF+SVD et Word2Vec)
```bash
python src/01_train_embeddings.py
```
> Entraîne les deux modèles à partir des PDFs dans `data/raw/Structured data/`.
> SentenceTransformers n'a pas besoin d'entraînement (modèle pré-entraîné).

### Étape 2 : Indexer les PDFs dans ChromaDB
```bash
python src/02_index_pdfs.py --embedding tfidf_svd --force
python src/02_index_pdfs.py --embedding word2vec --force
python src/02_index_pdfs.py --embedding sentence_transformer --force
```

### Étape 3 : Évaluer le retrieval (Hit@K, sans API)
```bash
python src/03_eval_retrieval.py --embedding tfidf_svd
python src/03_eval_retrieval.py --embedding word2vec
python src/03_eval_retrieval.py --embedding sentence_transformer
```

### Étape 4 : Évaluer l'agent end-to-end (utilise API Groq/Gemini)
```bash
python src/05_eval_agent.py --embedding tfidf_svd --limit 20
python src/05_eval_agent.py --embedding word2vec --limit 20
python src/05_eval_agent.py --embedding sentence_transformer --limit 20
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
├── 01_train_embeddings.py          # Entraîne TF-IDF+SVD et Word2Vec
├── 02_index_pdfs.py                # Indexe les PDFs dans ChromaDB
├── 03_eval_retrieval.py            # Évalue le retrieval (Hit@K)
├── 04_rag_agent.py                 # Agent RAG (Groq/Gemini)
└── 05_eval_agent.py                # Évaluation end-to-end
app.py                              # Interface web Flask
```

## Données

- `data/raw/Structured data/` : PDFs des rapports (source unique)
- `data/processed/data_ret_clean.csv` : Gabarito (questions + réponses attendues, pour évaluation uniquement)
- `.env` : Clés API (ignoré par Git)