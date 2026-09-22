import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import EmbeddingService
import numpy as np

# 1. Initialize the service
print("Loading model...")
service = EmbeddingService()
print(f"Model loaded! Vector dimension: {service.dimension}\n")

# 2. Three sentences to compare
q1 = "What is the return and refund policy?"
q2 = "Can I get my money back if I don't like the product?"
q3 = "The capital of France is Paris."

# 3. Generate vectors
v1 = np.array(service.embed_text(q1))
v2 = np.array(service.embed_text(q2))
v3 = np.array(service.embed_text(q3))

# 4. Cosine similarity function: (A . B) / (||A|| * ||B||)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Similarity between (Refund Policy) and (Money Back): {cosine_sim(v1, v2):.4f}")
print(f"Similarity between (Refund Policy) and (Paris):      {cosine_sim(v1, v3):.4f}")
