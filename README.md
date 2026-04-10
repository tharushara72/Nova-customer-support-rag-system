# 🌟 Nova — AI Customer Support Assistant

> An end-to-end **Retrieval Augmented Generation (RAG)** system for intelligent customer support automation. Built with a production-grade pipeline including semantic search, cross-encoder reranking, LLM generation, LLMOps observability, a REST API, and a chat UI — all containerized with Docker.

---

## 📋 Table of Contents

- [Demo](#demo)
- [RAG Architecture](#rag-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation — Without Docker](#installation--without-docker)
- [Installation — With Docker](#installation--with-docker)
- [LLMOps](#llmops)
- [API Reference](#api-reference)
- [Evaluation](#evaluation)

---

## 🎬 Demo

```
User: How do I get a refund?

Nova: To get a refund, you can start by reviewing our refund policy page
on our website. If you need further assistance, reach out to our support
team via live chat or phone. They will guide you through the process
including cancelling or adjusting your purchase.

⚡ 855ms  |  retrieval: 31ms  |  rerank: 267ms  |  generation: 557ms
```

---

## 🏗️ RAG Architecture

Nova uses a two-stage retrieval pipeline — retrieve broadly, then rank precisely.

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Embedding    │  ← all-MiniLM-L6-v2 (384-dim vector)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  FAISS Vector Search│  ← Semantic search → Top-10 candidates
│  (Bi-Encoder)       │    Fast but lower precision
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Cross-Encoder      │  ← Rerank Top-10 → Top-3
│  Reranker           │    Slower but much higher precision
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LLM Generation     │  ← Groq llama-3.3-70b-versatile
│  (with context)     │    Generates answer from retrieved context
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  LLMOps Layer       │  ← Logs latency, errors, Prometheus metrics
└─────────────────────┘
    │
    ▼
  Response
```

### Why Two-Stage Retrieval?

| Stage | Model Type | Speed | Accuracy | Why |
|---|---|---|---|---|
| FAISS search | Bi-encoder | ⚡ Very fast | Good | Pre-compute doc embeddings, query only at runtime |
| Cross-encoder reranker | Cross-encoder | Slower | Much better | Sees query + doc together, full attention |

A bi-encoder encodes the query and document **separately**, making it fast but slightly imprecise. A cross-encoder sees the query and document **together** as one input, which gives it full attention over both — much more accurate, but too slow to run over thousands of documents. The solution: use bi-encoder to retrieve top-10 quickly, then cross-encoder to rerank precisely to top-3. This is called **retrieve broadly → rank precisely**.

### Chunking Strategy

| Strategy | When to Use |
|---|---|
| Fixed-size with overlap | Long documents (PDFs, manuals) |
| Sentence-level | When semantic completeness per chunk matters |
| Q&A pair (used here) | When data is already structured as pairs |
| Recursive splitter | Mixed content |

Nova uses Q&A pair chunking because the customer support dataset is already structured as question-answer pairs. Each pair becomes one chunk, combining both the question and answer for richer retrieval context.

---

## 🛠️ Tech Stack

### Core Libraries

| Library | Version | Purpose | Why This Over Alternatives |
|---|---|---|---|
| `sentence-transformers` | ≥2.7.0 | Embedding model + cross-encoder reranker | Hugging Face native, pre-trained models, simple API. Better than raw `transformers` for sentence tasks. |
| `faiss-cpu` | ≥1.8.0 | Vector similarity search | Built by Meta AI, battle-tested. 10–100x faster than brute-force numpy. `IndexFlatIP` gives exact cosine search. Simpler than Pinecone/Weaviate for local use. |
| `openai` | ≥1.30.0 | LLM API client | OpenAI SDK works with Groq too (same interface, different `base_url`). No need for a separate Groq SDK. |
| `groq` (via openai SDK) | — | LLM inference | Free tier, very fast (700+ tokens/sec). Better for prototyping than paid OpenAI credits. |
| `fastapi` | ≥0.111.0 | REST API framework | Auto-generates Swagger UI. Async-native. Much faster than Flask. Type-safe with Pydantic. |
| `uvicorn` | ≥0.30.0 | ASGI server | The recommended server for FastAPI. Handles async requests efficiently. |
| `pydantic` | ≥2.7.0 | Data validation | FastAPI uses it natively. Validates request/response schemas automatically. |
| `prometheus-client` | ≥0.20.0 | Metrics collection | Industry standard. Integrates with Grafana dashboards. Better than custom logging for numeric metrics. |
| `python-dotenv` | ≥1.0.0 | Environment variable management | Keeps secrets out of code. Best practice for API keys. |
| `datasets` | ≥2.20.0 | Load HuggingFace datasets | One-line dataset loading. Automatic caching. |
| `streamlit` | latest | Chat UI | Zero-frontend-code needed. Fast to build. Great for demos and prototypes. |
| `pandas` | ≥2.2.0 | Data manipulation | Standard Python data library |
| `numpy` | ≥1.26.0 | Numerical operations | Required by FAISS and sentence-transformers |
| `tqdm` | ≥4.66.0 | Progress bars | Shows embedding progress during indexing |

### Models

| Model | Size | Purpose | Why |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22MB | Bi-encoder (embedding) | Fast, lightweight, 384-dim vectors. Great benchmark scores for semantic similarity. Good balance of speed vs quality. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90MB | Cross-encoder (reranker) | L-6 = only 6 layers → lower latency. Trained on MS MARCO passage ranking — ideal for Q&A reranking. |
| `llama-3.3-70b-versatile` | via API | Response generation | State-of-the-art open model. Free on Groq. Fast inference. |

### Infrastructure

| Tool | Purpose | Why This Over Alternatives |
|---|---|---|
| `uv` | Python package manager | 10–100x faster than pip. Automatic venv management. Lockfile for reproducibility. Modern replacement for pip + virtualenv + pip-tools. |
| `Docker` | Containerization | Reproducible environments. Works identically on any machine. Industry standard for deployment. |
| `Docker Compose` | Multi-container orchestration | Runs API + Prometheus together with one command. |
| `Prometheus` | Metrics scraping | Industry standard. Integrates with Grafana. Tracks latency, error rates, request counts. |

---

## 📁 Project Structure

```
nova-rag/
│
├── data/
│   └── prepare_data.py          # Load HuggingFace dataset, build chunks
│
├── indexing/
│   ├── __init__.py
│   ├── embedder.py              # Wraps all-MiniLM-L6-v2 embedding model
│   └── vector_store.py          # FAISS index: build, save, load, search
│
├── retrieval/
│   ├── __init__.py
│   ├── semantic_search.py       # Bi-encoder retrieval (top-10)
│   └── reranker.py              # Cross-encoder reranking (top-3)
│
├── generation/
│   ├── __init__.py
│   └── generator.py             # Prompt builder + Groq LLM call
│
├── llmops/
│   ├── __init__.py
│   ├── logger.py                # Structured JSON logging per request
│   └── metrics.py               # Prometheus counters and histograms
│
├── artifacts/                   # Auto-generated (gitignored)
│   ├── chunks.pkl               # Serialized document chunks
│   └── support_index.faiss      # FAISS vector index
│
├── logs/                        # Auto-generated (gitignored)
│   └── rag.log                  # Structured request logs
│
├── pipeline.py                  # Orchestrates all stages end-to-end
├── app.py                       # FastAPI application
├── ui.py                        # Streamlit chat interface
├── evaluate.py                  # RAG evaluation script
│
├── pyproject.toml               # Project metadata + dependencies (uv)
├── uv.lock                      # Exact dependency versions (commit this)
├── .env                         # API keys (never commit)
├── .gitignore
├── .dockerignore
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Multi-container setup
└── prometheus.yml               # Prometheus scrape config
```

---

## 💻 Installation — Without Docker

### Prerequisites

- Python 3.11 or higher
- `uv` package manager
- A Groq API key (free at https://console.groq.com)

### Step 1: Install uv

```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify
uv --version
```

### Step 2: Clone and set up the project

```bash
git clone https://github.com/YOUR_USERNAME/nova-rag.git
cd nova-rag

# Install all dependencies (creates .venv automatically)
uv sync
```

### Step 3: Set up environment variables

```bash
# Create .env file
echo "GROQ_API_KEY=your-groq-api-key-here" > .env
```

Get your free Groq API key at: https://console.groq.com/keys

### Step 4: Prepare data and build the index

```bash
# Download dataset and create chunks
uv run python data/prepare_data.py

# Build FAISS vector index
uv run python -c "
from data.prepare_data import load_and_prepare
from indexing.vector_store import VectorStore
chunks = load_and_prepare()
VectorStore().build(chunks)
"
```

This creates `artifacts/chunks.pkl` and `artifacts/support_index.faiss`.

### Step 5: Test the pipeline

```bash
uv run python pipeline.py
```

Expected output:
```
Pipeline ready.
Response: To get a refund, you can...
Total latency: 1302ms
Stage breakdown: {'retrieval_ms': 31.0, 'reranking_ms': 267.0, 'generation_ms': 1004.0}
```

### Step 6: Run the API

```bash
uv run uvicorn app:app --reload --port 8000
```

Visit: http://localhost:8000/docs

### Step 7: Run the Chat UI

Open a new terminal:

```bash
uv run streamlit run ui.py
```

Visit: http://localhost:8501

---

## 🐳 Installation — With Docker

### Prerequisites

- Docker Desktop installed and running
- A Groq API key (free at https://console.groq.com)

### Step 1: Clone the project

```bash
git clone https://github.com/YOUR_USERNAME/nova-rag.git
cd nova-rag
```

### Step 2: Set up environment variables

```bash
echo "GROQ_API_KEY=your-groq-api-key-here" > .env
```

### Step 3: Build the artifacts first (required before Docker)

The FAISS index must be built locally before running Docker, since it needs the dataset download:

```bash
# Install uv locally just for this step
pip install uv
uv sync
uv run python data/prepare_data.py
uv run python -c "
from data.prepare_data import load_and_prepare
from indexing.vector_store import VectorStore
chunks = load_and_prepare()
VectorStore().build(chunks)
"
```

### Step 4: Build and run with Docker Compose

```bash
# Build the Docker image
docker compose build

# Start all services (API + Prometheus)
docker compose up -d

# Check running containers
docker compose ps

# View logs
docker compose logs -f rag-api
```

### Step 5: Test

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I get a refund?"}'
```

### Step 6: Access services

| Service | URL | Description |
|---|---|---|
| Nova API | http://localhost:8000 | REST API |
| Swagger UI | http://localhost:8000/docs | Interactive API docs |
| Health Check | http://localhost:8000/health | Service status |
| Prometheus Metrics | http://localhost:8000/metrics | Raw metrics |
| Prometheus Dashboard | http://localhost:9090 | Metrics UI |

### Docker Commands Reference

```bash
# Stop all services
docker compose down

# Rebuild after code changes
docker compose down && docker compose build && docker compose up -d

# View resource usage
docker stats

# Shell into the container
docker exec -it customer-support-rag /bin/bash
```

---

## 📊 LLMOps

Nova includes production-grade observability covering three pillars: logging, metrics, and evaluation.

### Structured Logging

Every request is logged to `logs/rag.log` as structured JSON:

```json
{
  "timestamp": "2026-04-10T14:30:51.123456",
  "query": "How do I get a refund?",
  "stages_ms": {
    "retrieval_ms": 30.8,
    "reranking_ms": 267.0,
    "generation_ms": 557.5
  },
  "total_ms": 855.3,
  "response_preview": "To get a refund, you can start by reviewing...",
  "status": "success",
  "error": null
}
```

### Prometheus Metrics

Three key metrics are tracked and exposed at `/metrics`:

| Metric | Type | Description |
|---|---|---|
| `rag_requests_total` | Counter | Total number of requests |
| `rag_latency_seconds` | Histogram | End-to-end latency per request |
| `rag_errors_total` | Counter | Total number of failed requests |

Latency buckets: 100ms, 250ms, 500ms, 1s, 2s, 5s, 10s

### Viewing Metrics

```bash
# Raw Prometheus format
curl http://localhost:8000/metrics

# Prometheus UI (when running with Docker)
open http://localhost:9090
```

### Evaluation

Run the built-in evaluation suite:

```bash
uv run python evaluate.py
```

Sample output:
```
=== RAG Evaluation ===

[PASS] Query:    How do I cancel my order?
       Response:  I've come to understand that you would like to cancel...
       Latency:   1325ms

[PASS] Query:    I want a refund for my purchase
       Response:  I understand you are looking for a refund...
       Latency:   1956ms

[PASS] Query:    I cannot log into my account
       Response:  I've observed that you are having trouble logging in...
       Latency:   1257ms

[PASS] Query:    Where is my package?
       Response:  To find out where your package is...
       Latency:   919ms

[PASS] Query:    How do I update my billing information?
       Response:  To update your billing information, you can log into...
       Latency:   1047ms

========================================
Score:        5/5 (100%)
Avg latency:  1302ms
Max latency:  1957ms
Min latency:  920ms
========================================
```

### Performance Summary

| Stage | Avg Latency | Notes |
|---|---|---|
| FAISS retrieval | ~30ms | Pre-computed embeddings, very fast |
| Cross-encoder reranking | ~267ms | Runs on 10 candidates |
| LLM generation | ~1000ms | Network latency to Groq API |
| **Total** | **~1302ms** | End-to-end |

---

## 🔌 API Reference

### POST /ask

Submit a customer support query.

**Request:**
```json
{
  "query": "How do I get a refund?"
}
```

**Response:**
```json
{
  "query": "How do I get a refund?",
  "response": "To get a refund, you can start by reviewing our refund policy...",
  "total_latency_ms": 855.3,
  "stage_breakdown": {
    "retrieval_ms": 30.8,
    "reranking_ms": 267.0,
    "generation_ms": 557.5
  },
  "sources": [
    "I can't afford purchase {{Order Number}}",
    "I can no longer afford purchase {{Order Number}}",
    "want help cancelling purchase {{Order Number}}"
  ]
}
```

### GET /health

```json
{"status": "ok"}
```

### GET /metrics

Returns Prometheus-formatted metrics.

---

## 📈 Evaluation Results

| Query | Status | Latency |
|---|---|---|
| How do I cancel my order? | ✅ PASS | 1325ms |
| I want a refund for my purchase | ✅ PASS | 1956ms |
| I cannot log into my account | ✅ PASS | 1257ms |
| Where is my package? | ✅ PASS | 919ms |
| How do I update my billing information? | ✅ PASS | 1047ms |
| **Overall** | **5/5 (100%)** | **avg 1302ms** |

---

## 🙏 Acknowledgements

- Dataset: [Bitext Customer Support LLM Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Reranker: [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- LLM: [Llama 3.3 70B via Groq](https://console.groq.com)
