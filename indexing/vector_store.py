import faiss
import numpy as np
import pickle
import os
from indexing.embedder import Embedder

# Why FAISS?
# - Built by Meta AI, battle-tested in production
# - IndexFlatIP = exact cosine search (after L2 normalization)
# - For 5000 vectors, exact search is fast enough (<5ms)
# - For 1M+ vectors you'd switch to IndexIVFFlat (approximate)

ARTIFACTS_DIR = "artifacts"
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "support_index.faiss")
CHUNKS_PATH = os.path.join(ARTIFACTS_DIR, "chunks.pkl")


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]):
        """Build FAISS index from chunks."""
        embedder = Embedder()
        texts = [c["text"] for c in chunks]

        print("Generating embeddings...")
        embeddings = embedder.embed(texts)

        # IndexFlatIP = inner product (= cosine similarity when vectors are normalized)
        dimension = embeddings.shape[1]  # 384
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"Index built: {self.index.ntotal} vectors")

        # Save to disk
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        print("Saved index to disk.")

    def load(self):
        """Load existing index from disk."""
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded index: {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Return top_k most similar chunks."""
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            {"chunk": self.chunks[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
        ]