"""
02c_train_word2vec_pdf.py
Entraîne un modèle d'embedding TF-IDF + SVD à partir des PDFs du répertoire data/.

Le modèle est sauvegardé sous forme de fichier pickle dans :
    %LOCALAPPDATA%/rag-activeviam/models/word2vec_pdf.pkl

Note : le nom « word2vec » est conservé par convention du projet,
       mais la méthode réelle est TF-IDF + TruncatedSVD (pas gensim Word2Vec).
"""

import os
import pickle
import re
from pathlib import Path

import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models"
OUTPUT_PATH = OUTPUT_DIR / "word2vec_pdf.pkl"

# Dimension de la réduction SVD (300 pour cohérence avec sentence-transformers)
SVD_DIM = 300


# =========================
# Nettoyage de texte
# =========================

# Caractères Unicode privés et emojis à supprimer
_PRIVATE_USE_RE = re.compile(r"[\uf000-\uf0ff]")
# Guillemets typographiques -> ASCII
_FANCY_QUOTES = {
    "\u2018": "'", "\u2019": "'",  # guillemets simples
    "\u201c": '"', "\u201d": '"',  # guillemets doubles
}


def clean_text(text: str) -> str:
    """Nettoie un texte extrait de PDF : supprime les caractères privés, normalise les guillemets."""
    if not text:
        return ""
    # Supprimer les caractères de la zone privée Unicode
    text = _PRIVATE_USE_RE.sub("", text)
    # Normaliser les guillemets typographiques
    for old, new in _FANCY_QUOTES.items():
        text = text.replace(old, new)
    return text.strip()


# =========================
# Fonctions
# =========================

def extract_text_from_pdfs(data_dir: Path) -> list[str]:
    """Extrait le texte de tous les fichiers PDF dans data_dir (récursivement)."""
    texts = []

    # Chercher les PDFs dans tous les sous-répertoires
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"[ATTENTION] Aucun fichier PDF trouvé dans {data_dir}")
        return texts

    print(f"[INFO] {len(pdf_files)} fichiers PDF trouvés")

    for pdf_path in pdf_files:
        print(f"  Traitement de {pdf_path.name}...")
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(clean_text(text))
        except Exception as e:
            print(f"    [ERREUR] Échec de lecture de {pdf_path.name}: {e}")
            continue

    print(f"[INFO] {len(texts)} pages extraites des PDFs")
    return texts


def train_tfidf_svd_model(texts: list[str], output_path: Path) -> None:
    """Entraîne un modèle TF-IDF + SVD et le sauvegarde en pickle."""
    if not texts:
        raise ValueError("Aucun texte fourni pour l'entraînement")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Entraînement du vectoriseur TF-IDF sur {len(texts)} documents...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"[INFO] Matrice TF-IDF : {tfidf_matrix.shape}")

    print(f"[INFO] Réduction SVD (dimension = {SVD_DIM})...")
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    svd.fit(tfidf_matrix)
    print(f"[INFO] Variance expliquée (SVD) : {svd.explained_variance_ratio_.sum():.4f}")

    model_data = {
        "vectorizer": vectorizer,
        "svd": svd,
        "vector_size": SVD_DIM,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"[INFO] Modèle sauvegardé dans {output_path}")


# =========================
# Point d'entrée
# =========================

def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("Entraînement du modèle TF-IDF + SVD à partir des PDFs")
    print("=" * 70)

    # Extraire le texte des PDFs
    texts = extract_text_from_pdfs(DATA_DIR)

    if not texts:
        print("[ERREUR] Aucun texte PDF extrait. Impossible d'entraîner le modèle.")
        return

    # Entraîner et sauvegarder le modèle
    train_tfidf_svd_model(texts, OUTPUT_PATH)

    print("=" * 70)
    print("[OK] Entraînement terminé !")
    print("=" * 70)


if __name__ == "__main__":
    main()
