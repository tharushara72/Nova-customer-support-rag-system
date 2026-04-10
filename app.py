from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from pipeline import RAGPipeline
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import warnings
import logging
import os

# Suppress noisy warnings
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(
    title="Nova -Customer Support RAG",
    description="End-to-end RAG system with reranking and LLMOps",
    version="0.1.0",
)

# Load pipeline once when the app starts (not on every request)
print("Loading RAG pipeline...")
pipeline = RAGPipeline()
print("App ready!")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    response: str
    total_latency_ms: float
    stage_breakdown: dict
    sources: list[str]


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    result = pipeline.run(request.query)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)