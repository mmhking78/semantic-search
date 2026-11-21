import argparse
import numpy as np

from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(query, top_k):

    # Load saved embeddings
    corpus_embeddings = np.load("emb.npy")

    # Load original texts
    with open("corpus.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    # Convert query into embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Calculate cosine similarity
    scores = corpus_embeddings @ query_embedding

    # Get ranked results
    rank = np.argsort(-scores)

    # Print top-k results
    print("\nQuery:", query)
    print("\nTop", top_k, "Search Results")
    print("-" * 60)

    for i in rank[:top_k]:
        print(f"Score: {scores[i]:+.4f}")
        print(f"Text: {corpus[i]}")
        print("-" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Mini Semantic Search Engine"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Enter your search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    semantic_search(args.query, args.top_k)
