import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 24 short help-desk texts
sentences = [
    "How do I reset my password?",
    "I forgot my login password",
    "My account is locked",
    "I cannot access my account",
    "How can I change my username?",
    "How do I enable two factor authentication?",
    "My email verification is not working",
    "How can I recover my account?",
    "I need help with login problems",
    "How do I update my profile?",
    "My computer has a virus",
    "How do I configure a firewall?",
    "Firewall rules need updating",
    "How can I secure my network?",
    "My internet connection is slow",
    "How do I install antivirus software?",
    "I received a suspicious email",
    "How can I protect my personal data?",
    "My laptop is running slowly",
    "How do I update Windows?",
    "How to bake banana bread?",
    "What are the ingredients for chocolate cake?",
    "How do I make a vegetable soup?",
    "Give me tips for healthy cooking"
]

# Generate normalized embeddings
emb = model.encode(
    sentences,
    normalize_embeddings=True
)

print("Embedding Shape:", emb.shape)

# Save embeddings
np.save("emb.npy", emb)

print("Embeddings saved successfully to emb.npy")

# Save original texts
with open("corpus.txt", "w", encoding="utf-8") as f:
    for text in sentences:
        f.write(text + "\n")

print("Corpus saved successfully")
