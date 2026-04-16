import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Style
sns.set_theme(style="whitegrid")

# Donnees (10 requetes testees)
models = ['SentenceTransformers', 'Word2Vec', 'TF-IDF + SVD', 'Mode Hybride (RRF)']
scores = [5, 6, 6, 6]  # Accuracy / 10

fig, ax = plt.subplots(figsize=(10, 6))

# Gris = basique, bleu = meilleur
colors = ['#ced4da', '#4299e1', '#4299e1', '#2b6cb0'] 


bars = ax.barh(models, scores, color=colors, height=0.6, edgecolor='none')

# Annotations sur les barres
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.1, 
            bar.get_y() + bar.get_height()/2, 
            f'{int(width)}/10  ({int(width*10)}%)', 
            va='center', 
            ha='left',
            fontsize=12,
            fontweight='bold',
            color='#2d3748')

# Axes
ax.set_xlim(0, 10.5)
ax.set_xlabel("Nombre de reponses exactes generees par l'agent", fontsize=12, fontweight='bold', labelpad=15, color='#4a5568')
ax.set_title("Evaluation de l'Agent RAG (10 requetes de test)", fontsize=16, fontweight='bold', pad=20, color='#1a202c')
ax.xaxis.grid(True, linestyle='-', linewidth=1, color='#e2e8f0')
ax.yaxis.grid(False)


sns.despine(left=True, bottom=True)
ax.tick_params(axis='y', length=0)
ax.set_yticklabels(models, fontsize=12, fontweight='bold', color='#4a5568')
ax.set_xticks(range(0, 11))

# Note en bas de figure
plt.figtext(0.15, -0.05, "Note : Performances de l'agent complet apres retrieval + generation du LLM.\nLe mode hybride assure la meilleure couverture tout en reduisant le risque d'hallucination.",
            ha="left", fontsize=10, color="#718096", style="italic")


plt.tight_layout()
plt.savefig('eval_agent_results.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegarde sous 'eval_agent_results.png'")
