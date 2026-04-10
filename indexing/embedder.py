from sentence_transformers import SentenceTransformer
import numpy as np

# Why all-MiniLM-L6-v2?
# - Only 22MB — fast to load and run
# - 384-dimensional embeddings — compact but expressive
# - Excellent benchmark scores for semantic similarity
# - Much faster than larger models (bge-large, e5-large)
#   while giving ~90% of the accuracy

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Convert a list of strings into vectors.
        Returns shape: (len(texts), 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # needed for cosine similarity
        )
        return np.array(embeddings).astype("float32")