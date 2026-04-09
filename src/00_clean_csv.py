"""
00_clean_csv.py
Nettoie le CSV du dataset :
  - Supprime les caracteres Unicode de la zone privee (\\uf000-\\uf0ff)
  - Normalise les guillemets typographiques vers ASCII
  - Supprime les colonnes auto-generees (Unnamed)
  - Sauvegarde le CSV nettoye en remplacement

Utilisation :
  python src/00_clean_csv.py
"""

import re
import sys
from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "data_ret_clean.csv"

# Caracteres de la zone privee Unicode
_PRIVATE_USE_RE = re.compile(r"[\uf000-\uf0ff]")

# Guillemets typographiques -> ASCII
_REPLACEMENTS = {
    "\u2018": "'",   # guillemet simple gauche
    "\u2019": "'",   # guillemet simple droit
    "\u201c": '"',   # guillemet double gauche
    "\u201d": '"',   # guillemet double droit
    "\u2013": "-",   # tiret cadratin
    "\u2022": "-",   # puce
    "\u00a0": " ",   # espace insecable
}


# =========================
# Fonctions
# =========================

def clean_text(text: str) -> str:
    """Nettoie un texte individual."""
    if not isinstance(text, str) or not text.strip():
        return text

    # Supprimer les caracteres de la zone privee
    text = _PRIVATE_USE_RE.sub("", text)

    # Remplacer les caracteres typographiques
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)

    return text.strip()


def main():
    print(f"[INFO] Chargement du CSV : {CSV_PATH}")

    if not CSV_PATH.exists():
        print(f"[ERREUR] Fichier introuvable : {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] {len(df)} lignes, {len(df.columns)} colonnes")

    # Supprimer les colonnes auto-generees
    for col in list(df.columns):
        if col.lower().startswith("unnamed"):
            df = df.drop(columns=[col])
            print(f"[INFO] Colonne supprimee : {col}")

    # Compter les caracteres problematiques avant nettoyage
    text_cols = ["Question", "Context", "Value"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            n_private = df[col].apply(lambda x: bool(_PRIVATE_USE_RE.search(x))).sum()
            if n_private > 0:
                print(f"[INFO] {n_private} lignes avec caracteres prives dans '{col}'")

    # Nettoyer
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Sauvegarder
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"[OK] CSV nettoye sauvegarde : {CSV_PATH}")
    print(f"[INFO] {len(df)} lignes, colonnes : {list(df.columns)}")


if __name__ == "__main__":
    main()
