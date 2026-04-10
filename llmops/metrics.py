from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Why Prometheus?
# Industry standard for metrics. Integrates with Grafana dashboards.
# These three metrics cover: volume, latency, and reliability

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG pipeline requests"
)

LATENCY = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG pipeline latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total number of RAG pipeline errors"
)