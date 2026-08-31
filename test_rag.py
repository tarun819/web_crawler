"""Quick smoke test for Phase 6: Grounded RAG with Groq."""
import logging
from embeddings import ingest_pages, delete_collection
from retrieval import invalidate_bm25_cache
from rag import query_rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 1. Ingest test documentation
fake_pages = [
    {
        "url": "https://fastapi.example.com/tutorial",
        "domain": "fastapi.example.com",
        "title": "FastAPI First Steps",
        "text": (
            "FastAPI is a modern, fast web framework for building APIs with Python 3.8+ based on standard Python type hints.\n\n"
            "To define a path operation, use a decorator like @app.get('/items/{item_id}'). "
            "FastAPI automatically validates request data using Pydantic models."
        ),
        "depth": 0,
    },
    {
        "url": "https://fastapi.example.com/security",
        "domain": "fastapi.example.com",
        "title": "FastAPI Security & OAuth2",
        "text": (
            "FastAPI provides OAuth2PasswordBearer for bearer token authentication. "
            "Use Depends(oauth2_scheme) to protect API endpoints and extract security tokens from request headers."
        ),
        "depth": 0,
    },
]

result = ingest_pages(fake_pages)
invalidate_bm25_cache("fastapi.example.com")
print(f"✅ Ingested {result['chunks_stored']} chunks into ChromaDB\n")

# 2. Test RAG with a question present in docs
print("=" * 60)
print("TEST 1: In-domain question (Answerable)")
print("=" * 60)
q1 = "How does FastAPI validate request data?"
res1 = query_rag(q1, "fastapi.example.com")

print(f"Query:  {res1['query']}")
print(f"Answer:\n{res1['answer']}")
print(f"Sources: {[s['url'] for s in res1['sources']]}")
print(f"Latency: {res1['latency']}")

assert len(res1["answer"]) > 10, "Answer is too short"
assert "Pydantic" in res1["answer"] or "pydantic" in res1["answer"].lower(), "Answer should mention Pydantic"
print("✅ Grounded answer successfully generated!\n")

# 3. Test RAG with a question NOT present in docs (Should avoid hallucination)
print("=" * 60)
print("TEST 2: Out-of-domain question (Hallucination Prevention)")
print("=" * 60)
q2 = "What is the capital of France?"
res2 = query_rag(q2, "fastapi.example.com")

print(f"Query:  {res2['query']}")
print(f"Answer:\n{res2['answer']}")

# Check that the model adhered to grounding rules
lower_ans = res2["answer"].lower()
assert (
    "not have enough information" in lower_ans
    or "not mentioned" in lower_ans
    or "no information" in lower_ans
    or "does not contain" in lower_ans
), f"Model should decline to answer out-of-context question, but gave: {res2['answer']}"
print("✅ Hallucination prevention verified!")

# Cleanup
delete_collection("fastapi.example.com")
print("\n🎉 Phase 6 works!")
