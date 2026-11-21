from sentence_transformers import SentenceTransformer

# Load a small and fast embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into embedding
vec = model.encode("Stay Safe, Stay Secure")

print("Vector Type:", type(vec))
print("Vector Shape:", vec.shape)

# Display first 5 numbers
print("First 5 values:", vec[:5])
