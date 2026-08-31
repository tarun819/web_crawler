"""Quick smoke test for Phase 5: Hybrid Retrieval + RRF."""
import logging
from embeddings import ingest_pages, delete_collection
from retrieval import hybrid_search, invalidate_bm25_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. Ingest test pages with distinct keywords
fake_pages = [
    {
        "url": "https://test.example.com/asyncio",
        "domain": "test.example.com",
        "title": "Asyncio Guide",
        "text": "asyncio is a Python library for writing concurrent code using async and await syntax. It provides event loops, coroutines, and tasks for non-blocking I/O operations.",
        "depth": 0,
    },
    {
        "url": "https://test.example.com/httpx",
        "domain": "test.example.com",
        "title": "HTTPX Client",
        "text": "httpx.AsyncClient is an async HTTP client. Use httpx.TimeoutException to handle request timeouts. The httpx library supports HTTP/2 and connection pooling.",
        "depth": 0,
    },
    {
        "url": "https://test.example.com/testing",
        "domain": "test.example.com",
        "title": "Testing Guide",
        "text": "pytest is the most popular Python testing framework. Use fixtures for setup and teardown. Mocking replaces real objects with simulated ones during unit tests.",
        "depth": 0,
    },
]

result = ingest_pages(fake_pages)
invalidate_bm25_cache("test.example.com")
print(f"✅ Ingested {result['chunks_stored']} chunks\n")

# 2. Test hybrid search — semantic query
print("=" * 60)
print("TEST: Hybrid search for 'How to handle HTTP timeouts?'")
print("=" * 60)
search = hybrid_search("How to handle HTTP timeouts?", "test.example.com")

for r in search["results"]:
    print(f"  [{r['metadata']['title']}] RRF={r['rrf_score']:.4f} | vec_rank={r['vector_rank']} | bm25_rank={r['bm25_rank']}")
    print(f"    {r['text'][:80]}...")

assert len(search["results"]) > 0, "Search returned no results"
# Top result should be about httpx (contains "timeout")
top_title = search["results"][0]["metadata"]["title"]
assert "HTTPX" in top_title or "httpx" in search["results"][0]["text"].lower(), f"Expected httpx-related result, got: {top_title}"
print(f"\n✅ Top result is about '{top_title}' — correct!")
print(f"✅ Latency: {search['latency']}")

# Cleanup
delete_collection("test.example.com")
print("\n🎉 Phase 5 works!")
