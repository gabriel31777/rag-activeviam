"""
01_train_embeddings.py
Entraîne les modèles d'embedding locaux à partir des PDFs.

Deux modèles sont entraînés :
  1. TF-IDF + TruncatedSVD  ->  tfidf_svd_model.pkl
  2. Word2Vec (gensim)      ->  word2vec_model.pkl

Les modèles sont sauvegardés dans :
    %LOCALAPPDATA%/rag-activeviam/models/

Utilisation :
    python src/01_train_embeddings.py
    python src/01_train_embeddings.py --only tfidf_svd
    python src/01_train_embeddings.py --only word2vec
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import List

import fitz  # pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "Structured data"
MODELS_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models"

SVD_DIM = 300
W2V_DIM = 300

# Nettoyage de texte
_PRIVATE_USE_RE = re.compile(r"[\uf000-\uf0ff]")
_FANCY_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}


# =========================
# Extraction de texte
# =========================

def clean_text(text: str) -> str:
    """Nettoie un texte extrait de PDF."""
    if not text:
        return ""
    text = _PRIVATE_USE_RE.sub("", text)
    for old, new in _FANCY_QUOTES.items():
        text = text.replace(old, new)
    return text.strip()


def extract_pages_from_pdfs(pdf_dir: Path) -> List[str]:
    """Extrait le texte de chaque page de chaque PDF (via PyMuPDF).

    Retourne une liste de textes (un par page).
    """
    pages: List[str] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERREUR] Aucun PDF trouvé dans {pdf_dir}")
        return pages

    print(f"[INFO] {len(pdf_files)} fichiers PDF trouvés")

    for pdf_path in pdf_files:
        print(f"  Traitement de {pdf_path.name}...")
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                blocks = page.get_text("blocks")
                blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
                lines = []
                for b in blocks_sorted:
                    raw = b[4].strip()
                    if not raw:
                        continue
                    parts = [p.strip() for p in raw.split("\n") if p.strip()]
                    lines.append(" | ".join(parts) if len(parts) > 1 else parts[0])
                text = "\n".join(lines)
                if text.strip():
                    pages.append(clean_text(text))
            doc.close()
        except Exception as e:
            print(f"    [ERREUR] Échec de lecture de {pdf_path.name}: {e}")

    print(f"[INFO] {len(pages)} pages extraites")
    return pages


# =========================
# Entraînement TF-IDF + SVD
# =========================

def train_tfidf_svd(pages: List[str]) -> None:
    """Entraîne un modèle TF-IDF + SVD et le sauvegarde."""
    output_path = MODELS_DIR / "tfidf_svd_model.pkl"

    print(f"\n{'='*60}")
    print("Entraînement TF-IDF + SVD")
    print(f"{'='*60}")

    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=1,
        max_df=0.9,
        stop_words="english",
        sublinear_tf=True,  # Applique log(1 + tf) pour mieux capter les termes rares
    )
    tfidf_matrix = vectorizer.fit_transform(pages)
    print(f"[INFO] Matrice TF-IDF : {tfidf_matrix.shape}")

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    svd.fit(tfidf_matrix)
    print(f"[INFO] Variance expliquée : {svd.explained_variance_ratio_.sum():.4f}")

    model_data = {
        "vectorizer": vectorizer,
        "svd": svd,
        "vector_size": SVD_DIM,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"[OK] Modèle sauvegardé : {output_path}")


# =========================
# Entraînement Word2Vec
# =========================

def train_word2vec(pages: List[str]) -> None:
    """Entraîne un vrai modèle Word2Vec (gensim) et le sauvegarde."""
    from gensim.models import Word2Vec

    output_path = MODELS_DIR / "word2vec_model.pkl"

    print(f"\n{'='*60}")
    print("Entraînement Word2Vec (gensim)")
    print(f"{'='*60}")

    # Tokeniser chaque page en liste de mots
    sentences = [page.lower().split() for page in pages]
    print(f"[INFO] {len(sentences)} documents, {sum(len(s) for s in sentences)} tokens")

    # Entraîner Word2Vec
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=W2V_DIM,
        window=5,
        min_count=2,
        workers=4,
        epochs=20,
        sg=1,  # Skip-gram (meilleur pour petits corpus)
    )
    print(f"[INFO] Vocabulaire Word2Vec : {len(w2v_model.wv)} mots")

    # Entraîner aussi un TF-IDF pour les poids de la moyenne pondérée
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=1,
        stop_words="english",
    )
    vectorizer.fit(pages)

    model_data = {
        "w2v_model": w2v_model,
        "vectorizer": vectorizer,
        "vector_size": W2V_DIM,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"[OK] Modèle sauvegardé : {output_path}")


# =========================
# Point d'entrée
# =========================

def main():
    ap = argparse.ArgumentParser(description="Entraîner les modèles d'embedding")
    ap.add_argument(
        "--only", choices=["tfidf_svd", "word2vec"],
        default=None, help="Entraîner un seul type de modèle",
    )
    args = ap.parse_args()

    if not PDF_DIR.exists():
        print(f"[ERREUR] Répertoire PDF introuvable : {PDF_DIR}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Extraire le texte de tous les PDFs
    pages = extract_pages_from_pdfs(PDF_DIR)
    if not pages:
        print("[ERREUR] Aucun texte extrait. Impossible d'entraîner.")
        return

    if args.only is None or args.only == "tfidf_svd":
        train_tfidf_svd(pages)

    if args.only is None or args.only == "word2vec":
        train_word2vec(pages)

    print(f"\n{'='*60}")
    print("[OK] Entraînement terminé !")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
