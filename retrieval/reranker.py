from sentence_transformers import CrossEncoder

# Why cross-encoder vs bi-encoder?
#
# Bi-encoder (FAISS step):
#   - Encodes query and document SEPARATELY
#   - Fast: can pre-compute doc embeddings
#   - Less accurate: no direct query-doc interaction
#
# Cross-encoder (this step):
#   - Takes [query + document] as ONE input
#   - Slow: can't pre-compute, must run for each pair
#   - Much more accurate: full attention across both
#
# Why 'ms-marco-MiniLM-L-6-v2'?
#   - L-6 = only 6 transformer layers (lightweight → lower latency)
#   - Trained on MS MARCO passage ranking (ideal for Q&A)
#   - Much faster than L-12 with ~85% of the accuracy

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
        """
        Score each (query, passage) pair and return top_n.
        """
        pairs = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )

        return [
            {"chunk": c["chunk"], "rerank_score": float(s)}
            for s, c in ranked[:top_n]
        ]