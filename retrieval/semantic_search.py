import numpy as np
import faiss
from indexing.embedder import Embedder
from indexing.vector_store import VectorStore


class SemanticSearch:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.store.load()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Embed the query and find top_k nearest chunks.

        Why top_k=10 and not 3?
        We retrieve more than we need because:
        1. The bi-encoder (this step) is fast but imprecise
        2. The cross-encoder (next step) is precise but slow
        3. We give the reranker more candidates to work with
        This pattern is called: retrieve broadly → rank precisely
        """
        query_embedding = self.embedder.embed([query])
        faiss.normalize_L2(query_embedding)
        return self.store.search(query_embedding, top_k)