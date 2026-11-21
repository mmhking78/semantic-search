import numpy as np

def cosine(a, b):
    return float(
        a @ b /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )

# Load saved embeddings
emb = np.load("emb.npy")

labels = [
    "reset password",
    "forgot login",
    "locked account",
    "firewall",
    "banana bread"
]

# Compare first 5 embeddings
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        s = cosine(emb[i], emb[j])

        print(
            f"{labels[i]:18} vs "
            f"{labels[j]:18} {s:+.2f}"
        )
