"""
RAG Evaluation and Quantitative Benchmarking Framework.

Measures:
1. Retrieval Metrics: Hit Rate @ K (Hit@1, Hit@3, Hit@5) and Mean Reciprocal Rank (MRR).
2. Strategy Comparison: Vector-Only vs. BM25-Only vs. Hybrid Search (RRF).
3. Latency Benchmarks: Mean, P50, and P95 retrieval latency.
4. RAG Generation Quality: Groundedness, Citation Precision, and Generation Latency.
"""
import logging
import statistics
import time
from typing import Callable

import config
from embeddings import (
    ingest_pages,
    delete_collection,
    get_or_create_collection,
)
from retrieval import vector_search, bm25_search, hybrid_search, invalidate_bm25_cache
from rag import query_rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_DOMAIN = "benchmark.eval.internal"


# Synthetic Gold-Standard Evaluation Corpus

EVAL_PAGES = [
    {
        "url": f"https://{BENCHMARK_DOMAIN}/asyncio-basics",
        "domain": BENCHMARK_DOMAIN,
        "title": "Asyncio Coroutines and Tasks",
        "text": (
            "asyncio is a library to write concurrent code using the async/await syntax. "
            "To run multiple coroutines concurrently, use asyncio.gather(*tasks). "
            "asyncio.create_task() schedules a coroutine for execution in the event loop. "
            "Use asyncio.sleep(delay) to pause execution asynchronously without blocking the OS thread."
        ),
        "depth": 0,
    },
    {
        "url": f"https://{BENCHMARK_DOMAIN}/httpx-client",
        "domain": BENCHMARK_DOMAIN,
        "title": "HTTPX Async Client & Timeouts",
        "text": (
            "HTTPX provides httpx.AsyncClient for asynchronous HTTP requests. "
            "Configure timeouts using httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0). "
            "Catch httpx.TimeoutException to handle network timeouts gracefully. "
            "httpx.HTTPStatusError is raised when response.raise_for_status() encounters 4xx or 5xx codes."
        ),
        "depth": 0,
    },
    {
        "url": f"https://{BENCHMARK_DOMAIN}/fastapi-security",
        "domain": BENCHMARK_DOMAIN,
        "title": "FastAPI OAuth2 & Bearer Authentication",
        "text": (
            "FastAPI provides OAuth2PasswordBearer to extract bearer tokens from Authorization headers. "
            "Declare oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token'). "
            "Inject authentication into routes using token: str = Depends(oauth2_scheme). "
            "Pass HTTP 401 HTTPException with WWW-Authenticate header on invalid credentials."
        ),
        "depth": 0,
    },
    {
        "url": f"https://{BENCHMARK_DOMAIN}/chromadb-vector",
        "domain": BENCHMARK_DOMAIN,
        "title": "ChromaDB HNSW Indexing & Cosine Distance",
        "text": (
            "ChromaDB stores embeddings with an in-memory or persisted HNSW vector index. "
            "Configure distance metric using metadata={'hnsw:space': 'cosine'}. "
            "Collection.upsert() performs idempotent updates based on deterministic document IDs. "
            "Query nearest neighbors with collection.query(query_embeddings=[...], n_results=10)."
        ),
        "depth": 0,
    },
    {
        "url": f"https://{BENCHMARK_DOMAIN}/bm25-ranking",
        "domain": BENCHMARK_DOMAIN,
        "title": "BM25Okapi Keyword Retrieval & Tokenization",
        "text": (
            "BM25Okapi ranks documents based on term frequency (TF) and inverse document frequency (IDF). "
            "It penalizes document length with parameters k1=1.5 and b=0.75. "
            "Exact keyword matching ensures rare technical symbols like 'OAuth2PasswordBearer' are found. "
            "Combine BM25 and Vector search using Reciprocal Rank Fusion with constant k=60."
        ),
        "depth": 0,
    },
]


# Evaluation Queries with Ground-Truth Target URLs

EVAL_DATASET = [
    # 1. Exact Keyword Technical Query (BM25 advantage)
    {
        "query": "How to catch TimeoutException in httpx?",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/httpx-client",
        "expected_fact": "TimeoutException",
        "query_type": "Keyword-heavy",
    },
    # 2. Semantic Conceptual Query (Vector advantage)
    {
        "query": "How to run multiple coroutines in parallel without blocking?",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/asyncio-basics",
        "expected_fact": "asyncio.gather",
        "query_type": "Semantic/Conceptual",
    },
    # 3. Exact Symbol Query (BM25 advantage)
    {
        "query": "OAuth2PasswordBearer bearer token dependency",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/fastapi-security",
        "expected_fact": "OAuth2PasswordBearer",
        "query_type": "Exact Symbol",
    },
    # 4. Semantic Vector Indexing Query
    {
        "query": "Configuring cosine similarity space and nearest neighbor search in database",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/chromadb-vector",
        "expected_fact": "cosine",
        "query_type": "Semantic/Conceptual",
    },
    # 5. Hybrid Keyword + Concept Query
    {
        "query": "Term frequency inverse document frequency rank fusion k=60",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/bm25-ranking",
        "expected_fact": "Reciprocal Rank Fusion",
        "query_type": "Hybrid/Complex",
    },
    # 6. Error handling / Status Codes
    {
        "query": "What exception is raised on 4xx or 5xx response in http client?",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/httpx-client",
        "expected_fact": "HTTPStatusError",
        "query_type": "Semantic + Keyword",
    },
    # 7. Authentication Token Injection
    {
        "query": "How to extract token using Depends in API route header?",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/fastapi-security",
        "expected_fact": "Depends(oauth2_scheme)",
        "query_type": "Semantic/Conceptual",
    },
    # 8. Event loop scheduling
    {
        "query": "How to schedule coroutine execution in event loop?",
        "expected_url": f"https://{BENCHMARK_DOMAIN}/asyncio-basics",
        "expected_fact": "asyncio.create_task",
        "query_type": "Semantic/Conceptual",
    },
]


# Strategy Wrappers for Evaluation

def search_vector_only(query: str, domain: str) -> list[dict]:
    """Pure Vector Search baseline."""
    return vector_search(query, domain, top_k=5)


def search_bm25_only(query: str, domain: str) -> list[dict]:
    """Pure BM25 Keyword Search baseline."""
    return bm25_search(query, domain, top_k=5)


def search_hybrid_rrf(query: str, domain: str) -> list[dict]:
    """Hybrid Search (Vector + BM25 + RRF)."""
    res = hybrid_search(query, domain)
    return res.get("results", [])


# Retrieval Metric Evaluator

def evaluate_strategy(
    strategy_name: str,
    search_fn: Callable[[str, str], list[dict]],
    dataset: list[dict],
    domain: str,
) -> dict:
    """
    Evaluates Hit@1, Hit@3, Hit@5, MRR, and Latency for a search strategy.
    """
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []
    latencies_ms = []

    for item in dataset:
        q = item["query"]
        target_url = item["expected_url"]

        start = time.perf_counter()
        results = search_fn(q, domain)
        latency = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency)

        # Check rankings
        found_rank = None
        for rank_idx, doc in enumerate(results, start=1):
            doc_url = doc.get("metadata", {}).get("url", "")
            if doc_url == target_url:
                found_rank = rank_idx
                break

        if found_rank == 1:
            hit_at_1 += 1
        if found_rank and found_rank <= 3:
            hit_at_3 += 1
        if found_rank and found_rank <= 5:
            hit_at_5 += 1

        rr = (1.0 / found_rank) if found_rank else 0.0
        reciprocal_ranks.append(rr)

    n = len(dataset)
    return {
        "strategy": strategy_name,
        "hit_at_1": hit_at_1 / n,
        "hit_at_3": hit_at_3 / n,
        "hit_at_5": hit_at_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": sorted(latencies_ms)[int(n * 0.95)] if n > 1 else latencies_ms[0],
    }


# End-to-End RAG Generation Evaluator

def evaluate_rag_pipeline(dataset: list[dict], domain: str) -> dict:
    """
    Evaluates End-to-End RAG generation: Answer Groundedness, Citation Precision, and Latency.
    """
    grounded_correct = 0
    citations_valid = 0
    retrieval_latencies = []
    llm_latencies = []
    total_latencies = []

    for item in dataset:
        q = item["query"]
        expected_fact = item["expected_fact"]
        target_url = item["expected_url"]

        res = query_rag(q, domain)
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        latency = res.get("latency", {})

        # Fact Grounding Check: Is the key factual concept present in the answer?
        if expected_fact.lower() in answer.lower():
            grounded_correct += 1

        # Citation Validity Check: Did the LLM cite the target ground truth URL?
        cited_urls = [s.get("url") for s in sources]
        if target_url in cited_urls:
            citations_valid += 1

        retrieval_latencies.append(latency.get("retrieval_ms", 0.0))
        llm_latencies.append(latency.get("llm_ms", 0.0))
        total_latencies.append(latency.get("total_ms", 0.0))

    n = len(dataset)
    return {
        "grounded_accuracy": grounded_correct / n,
        "citation_precision": citations_valid / n,
        "mean_retrieval_ms": statistics.mean(retrieval_latencies),
        "mean_llm_ms": statistics.mean(llm_latencies),
        "mean_total_ms": statistics.mean(total_latencies),
    }


# Formatting Benchmark Report

def print_benchmark_report(retrieval_results: list[dict], rag_results: dict):
    """Prints a beautiful markdown table and summary of benchmark results."""
    print("\n" + "=" * 80)
    print("🏆 RAG SEARCH ENGINE QUANTITATIVE BENCHMARK REPORT")
    print("=" * 80)

    print("\n### 1. Retrieval Strategy Comparison")
    print("| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Mean Latency | P95 Latency |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in retrieval_results:
        print(
            f"| **{r['strategy']}** | "
            f"{r['hit_at_1']*100:.1f}% | "
            f"{r['hit_at_3']*100:.1f}% | "
            f"{r['hit_at_5']*100:.1f}% | "
            f"**{r['mrr']:.4f}** | "
            f"{r['latency_mean_ms']:.1f} ms | "
            f"{r['latency_p95_ms']:.1f} ms |"
        )

    print("\n### 2. End-to-End RAG Generation Metrics (Groq LLM)")
    print(f"• **Factual Grounding Accuracy:** {rag_results['grounded_accuracy']*100:.1f}%")
    print(f"• **Citation Precision:**        {rag_results['citation_precision']*100:.1f}%")
    print(f"• **Mean Retrieval Latency:**    {rag_results['mean_retrieval_ms']:.1f} ms")
    print(f"• **Mean LLM Generation Time:**  {rag_results['mean_llm_ms']:.1f} ms")
    print(f"• **Total End-to-End Latency:**  {rag_results['mean_total_ms']:.1f} ms")
    print("=" * 80 + "\n")


def run_benchmark():
    """Main benchmark execution workflow."""
    print("\n[1/4] Ingesting synthetic gold-standard corpus into ChromaDB...")
    ingest_pages(EVAL_PAGES)
    invalidate_bm25_cache(BENCHMARK_DOMAIN)

    print(f"[2/4] Evaluating {len(EVAL_DATASET)} queries across 3 retrieval strategies...")
    vector_metrics = evaluate_strategy(
        "Vector Only (BGE-small)",
        search_vector_only,
        EVAL_DATASET,
        BENCHMARK_DOMAIN,
    )
    bm25_metrics = evaluate_strategy(
        "BM25 Only (Okapi)",
        search_bm25_only,
        EVAL_DATASET,
        BENCHMARK_DOMAIN,
    )
    hybrid_metrics = evaluate_strategy(
        "Hybrid (Vector + BM25 + RRF)",
        search_hybrid_rrf,
        EVAL_DATASET,
        BENCHMARK_DOMAIN,
    )

    retrieval_results = [vector_metrics, bm25_metrics, hybrid_metrics]

    print("[3/4] Evaluating End-to-End RAG generation with Groq...")
    rag_metrics = evaluate_rag_pipeline(EVAL_DATASET, BENCHMARK_DOMAIN)

    print("[4/4] Cleaning up benchmark collection...")
    delete_collection(BENCHMARK_DOMAIN)

    print_benchmark_report(retrieval_results, rag_metrics)


if __name__ == "__main__":
    run_benchmark()
