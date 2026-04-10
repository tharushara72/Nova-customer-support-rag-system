import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pipeline import RAGPipeline

TEST_CASES = [
    {
        "query": "How do I cancel my order?",
        "expected_keywords": ["cancel", "order"],
    },
    {
        "query": "I want a refund for my purchase",
        "expected_keywords": ["refund", "return"],
    },
    {
        "query": "I cannot log into my account",
        "expected_keywords": ["account", "login", "password"],
    },
    {
        "query": "Where is my package?",
        "expected_keywords": ["delivery", "track", "shipment"],
    },
    {
        "query": "How do I update my billing information?",
        "expected_keywords": ["billing", "payment", "update"],
    },
]


def evaluate():
    pipeline = RAGPipeline()
    print("\n=== RAG Evaluation ===\n")

    passed = 0
    total = len(TEST_CASES)
    latencies = []

    for tc in TEST_CASES:
        result = pipeline.run(tc["query"])
        response = result["response"].lower()
        latency = result["total_latency_ms"]
        latencies.append(latency)

        hit = any(kw in response for kw in tc["expected_keywords"])
        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1

        print(f"[{status}] Query:    {tc['query']}")
        print(f"       Response:  {result['response'][:120]}...")
        print(f"       Latency:   {latency}ms")
        print()

    print("=" * 40)
    print(f"Score:        {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Avg latency:  {sum(latencies)/len(latencies):.0f}ms")
    print(f"Max latency:  {max(latencies):.0f}ms")
    print(f"Min latency:  {min(latencies):.0f}ms")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()