import time
import os
from retrieval.semantic_search import SemanticSearch
from retrieval.reranker import Reranker
from generation.generator import Generator
from llmops.logger import log_pipeline
from llmops.metrics import REQUEST_COUNT, LATENCY, ERROR_COUNT


class RAGPipeline:
    def __init__(self):
        print("Initializing RAG pipeline...")
        self.searcher = SemanticSearch()
        self.reranker = Reranker()
        self.generator = Generator()
        print("Pipeline ready.")

    def run(self, query: str) -> dict:
        REQUEST_COUNT.inc()
        stages = {}
        start = time.time()

        try:
            # Stage 1: Semantic Search
            t0 = time.time()
            candidates = self.searcher.search(query, top_k=10)
            stages["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

            # Stage 2: Reranking
            t0 = time.time()
            reranked = self.reranker.rerank(query, candidates, top_n=3)
            stages["reranking_ms"] = round((time.time() - t0) * 1000, 1)

            # Stage 3: Generation
            t0 = time.time()
            response = self.generator.generate(query, reranked)
            stages["generation_ms"] = round((time.time() - t0) * 1000, 1)

            total = round((time.time() - start) * 1000, 1)
            LATENCY.observe(total / 1000)

            log_pipeline(query, stages, response)

            return {
                "query": query,
                "response": response,
                "total_latency_ms": total,
                "stage_breakdown": stages,
                "sources": [r["chunk"]["question"] for r in reranked],
            }

        except Exception as e:
            ERROR_COUNT.inc()
            log_pipeline(query, stages, None, status="error", error=str(e))
            raise e


# Quick test
if __name__ == "__main__":
    pipeline = RAGPipeline()
    result = pipeline.run("How do I cancel my subscription?")
    print(f"\nResponse: {result['response']}")
    print(f"Total latency: {result['total_latency_ms']}ms")
    print(f"Stage breakdown: {result['stage_breakdown']}")