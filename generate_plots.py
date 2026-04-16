import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Style
plt.style.use('ggplot')
sns.set_context("talk")
sns.set_palette("crest")

# ===============
# FIG 1 : Hit@10
# ===============
models = ['Word2Vec (Gensim)', 'SentenceTransformers', 'TF-IDF + SVD', 'Hybride (RRF)']
scores = [62.0, 66.2, 69.0, 69.4]
colors = ['#8ca2c4', '#688cb6', '#4775a6', '#214e7a']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(models, scores, color=colors, edgecolor='black', linewidth=1.2)

# Annotations
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width}%', 
            va='center', ha='left', fontsize=12, fontweight='bold')

ax.set_xlim(55, 75)
ax.set_xlabel("Hit@10 Accuracy (%)", fontweight='bold')
ax.set_title("Performance du Retrieval par Algorithme (Hit@10)", fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("retrieval_performance.png", dpi=300)
plt.close()

# ===============
# FIG 2 : Semantics vs Lexical Trade-off
# ===============
fig, ax = plt.subplots(figsize=(9, 7))

# X: Semantic Capability (0-10)
# Y: Exact Literal Lexical Match (0-10)
data = {
    'Word2Vec': (4.0, 3.5, '#8ca2c4'),
    'SentenceTransformers': (9.0, 5.0, '#688cb6'),
    'TF-IDF + SVD': (3.0, 9.5, '#4775a6'),
    'Hybride (RRF)': (9.0, 9.5, '#e74c3c') # Red for hybrid
}

ax.grid(True, linestyle='--', alpha=0.6)

for name, (x, y, color) in data.items():
    ax.scatter(x, y, s=800, color=color, edgecolor='black', zorder=5, label=name)
    ax.text(x, y + 0.4, name, fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=6)

ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.set_xlabel("Capacite Semantique (Generalisation)", fontweight='bold', fontsize=13)
ax.set_ylabel("Precision Lexicale Exacte (Taches Tabulaires)", fontweight='bold', fontsize=13)
ax.set_title("Matrice de Qualite des Algorithmes (Theorique)", fontsize=16, fontweight='bold', pad=20)

# Lignes de separation
ax.axhline(5, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.axvline(5, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("algo_tradeoff.png", dpi=300)
plt.close()

print("Graphiques generes : retrieval_performance.png et algo_tradeoff.png")
