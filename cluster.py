import numpy as np

from sklearn.cluster import KMeans

# Load embeddings
emb = np.load("emb.npy")

# Create 4 clusters
k = 4

model = KMeans(
    n_clusters=k,
    n_init=10,
    random_state=0
)

labels = model.fit_predict(emb)

# Load original texts
with open("corpus.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# Display clusters
for text, cluster in zip(texts, labels):
    print(f"Cluster {cluster}: {text}")
