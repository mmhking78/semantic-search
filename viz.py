import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Load embeddings
emb = np.load("emb.npy")

# Load original texts
with open("corpus.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Reduce 384 dimensions to 2 dimensions
pca = PCA(n_components=2)

xy = pca.fit_transform(emb)

# Create scatter plot
plt.figure(figsize=(14, 9))

plt.scatter(
    xy[:, 0],
    xy[:, 1]
)

# Add text labels
for (x, y), text in zip(xy, texts):
    plt.annotate(
        text[:28],
        (x, y),
        fontsize=8
    )

plt.title("AI LLM Class 11 - Semantic Embeddings")

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid(True)

plt.tight_layout()

# Save image
plt.savefig(
    "embeddings.png",
    dpi=150
)

print("2D Embedding Plot saved as embeddings.png")

plt.show()
