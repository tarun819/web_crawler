"""
FastAPI backend for the Web Crawler + Hybrid RAG Search Engine.

Endpoints:
  POST /crawl   — Crawl a public URL, chunk text, embed into ChromaDB.
  POST /search  — Hybrid search (Vector + BM25 + RRF) returning raw chunks.
  POST /query   — Grounded RAG: retrieves chunks + generates LLM answer with citations.
  GET  /health  — Health check with uptime and active collections.

All endpoints are rate-limited per client IP (sliding window, in-memory).
"""
import logging
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from crawler import crawl
from embeddings import ingest_pages, list_collections, get_or_create_collection
from retrieval import hybrid_search, invalidate_bm25_cache
from rag import query_rag
from ratelimit import check_rate_limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Track server start time for /health uptime reporting
SERVER_START_TIME = time.monotonic()

app = FastAPI(
    title="Documentation RAG Search Engine",
    description="Crawl documentation sites, embed chunks into ChromaDB, and query with grounded RAG.",
    version="1.0.0",
)

# CORS: allow Gradio UI and local development to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request/Response Schemas

class CrawlRequest(BaseModel):
    url: str = Field(..., description="Public URL to crawl (must pass SSRF validation)")
    max_pages: int = Field(default=20, ge=1, le=50, description="Maximum pages to crawl")
    max_depth: int = Field(default=2, ge=0, le=5, description="Maximum BFS crawl depth")

class CrawlResponse(BaseModel):
    domain: str
    pages_crawled: int
    chunks_stored: int
    collection_name: str

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    domain: str = Field(..., description="Domain to search (e.g., docs.python.org)")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Question to answer")
    domain: str = Field(..., description="Domain to query against")


# Rate Limit Middleware

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply per-IP sliding-window rate limiting to all endpoints."""
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    is_limited, remaining = check_rate_limit(client_ip)

    if is_limited:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again in 60 seconds."},
        )

    response = await call_next(request)
    return response


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check with uptime and active collections."""
    uptime_seconds = round(time.monotonic() - SERVER_START_TIME, 1)
    collections = list_collections()
    return {
        "status": "healthy",
        "uptime_seconds": uptime_seconds,
        "active_collections": len(collections),
        "collections": collections,
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_endpoint(request: CrawlRequest):
    """
    Crawl a public URL, extract text, chunk it, embed into ChromaDB.

    Returns a summary of pages crawled and chunks stored.
    """
    logger.info(f"Crawl request: url={request.url}, max_pages={request.max_pages}, max_depth={request.max_depth}")

    try:
        # Phase 2: Crawl the site
        pages = await crawl(
            seed_url=request.url,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
        )

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Crawl returned no pages. The URL may be blocked by robots.txt, "
                       "unreachable, or a private/local address.",
            )

        # Phase 3+4: Chunk and ingest into ChromaDB
        ingest_result = ingest_pages(pages)

        # Invalidate BM25 cache so next search picks up new chunks
        invalidate_bm25_cache(ingest_result["domain"])

        logger.info(f"Crawl complete: {ingest_result['pages_ingested']} pages, {ingest_result['chunks_stored']} chunks")

        return CrawlResponse(
            domain=ingest_result["domain"],
            pages_crawled=ingest_result["pages_ingested"],
            chunks_stored=ingest_result["chunks_stored"],
            collection_name=ingest_result["collection_name"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Crawl failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Hybrid search: Vector + BM25 + RRF fusion.

    Returns top-k ranked chunks with metadata and latency breakdown.
    """
    logger.info(f"Search request: query='{request.query}', domain='{request.domain}'")

    # Verify domain has been crawled
    collection = get_or_create_collection(request.domain)
    if collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No crawled data found for domain '{request.domain}'. Please crawl it first via POST /crawl.",
        )

    result = hybrid_search(request.query, request.domain)

    # Clean results for JSON response (remove non-serializable fields)
    clean_results = []
    for r in result.get("results", []):
        clean_results.append({
            "text": r["text"],
            "metadata": r["metadata"],
            "rrf_score": round(r["rrf_score"], 4),
            "vector_rank": r.get("vector_rank"),
            "bm25_rank": r.get("bm25_rank"),
        })

    return {
        "query": result["query"],
        "domain": result["domain"],
        "results": clean_results,
        "latency": result["latency"],
    }


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Grounded RAG: retrieves relevant chunks and generates an LLM answer with citations.

    Returns the generated answer, source citations, and latency breakdown.
    """
    logger.info(f"RAG query: query='{request.query}', domain='{request.domain}'")

    # Verify domain has been crawled
    collection = get_or_create_collection(request.domain)
    if collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No crawled data found for domain '{request.domain}'. Please crawl it first via POST /crawl.",
        )

    result = query_rag(request.query, request.domain)

    return {
        "query": result["query"],
        "domain": result["domain"],
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
        "latency": result["latency"],
    }
