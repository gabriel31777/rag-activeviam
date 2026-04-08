"""
02c_train_word2vec_pdf.py
- Treina um modelo Word2Vec (TF-IDF + SVD) a partir dos PDFs no diretório data/
- Salva o modelo em %LOCALAPPDATA%/rag-activeviam/models/word2vec_pdf.pkl
"""

import os
import pickle
from pathlib import Path
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# =========================
# Config
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "rag-activeviam" / "models"
OUTPUT_PATH = OUTPUT_DIR / "word2vec_pdf.pkl"

# SVD dimensionality (same as sentence-transformers embedding size for consistency)
SVD_DIM = 300

# =========================
# Functions
# =========================

def extract_text_from_pdfs(data_dir: Path) -> list[str]:
    """Extract all text from PDF files in data_dir."""
    texts = []
    
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDF files found in {data_dir}")
        return texts
    
    print(f"[INFO] Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"  Processing {pdf_path.name}...")
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
        except Exception as e:
            print(f"    [ERROR] Failed to read {pdf_path.name}: {e}")
            continue
    
    print(f"[INFO] Extracted {len(texts)} pages from PDFs")
    return texts

def train_word2vec_model(texts: list[str], output_path: Path) -> None:
    """Train TF-IDF + SVD model and save it."""
    
    if not texts:
        raise ValueError("No texts provided for training")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Training TF-IDF vectorizer on {len(texts)} documents...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"[INFO] TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    print(f"[INFO] Training SVD (dimensionality reduction to {SVD_DIM})...")
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    svd.fit(tfidf_matrix)
    print(f"[INFO] SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    model_data = {
        'vectorizer': vectorizer,
        'svd': svd,
        'vector_size': SVD_DIM
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"[INFO] Model saved to {output_path}")

def main():
    """Main entry point."""
    print("=" * 70)
    print("Training Word2Vec (TF-IDF + SVD) model from PDFs")
    print("=" * 70)
    
    # Extract text from PDFs
    texts = extract_text_from_pdfs(DATA_DIR)
    
    if not texts:
        print("[ERROR] No PDF texts extracted. Cannot train model.")
        return
    
    # Train and save model
    train_word2vec_model(texts, OUTPUT_PATH)
    
    print("=" * 70)
    print("✅ Training complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
