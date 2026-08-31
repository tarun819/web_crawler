"""
Test suite for Phase 4: Embeddings + ChromaDB Storage.

Verifies:
1. Embedding model loads and produces 384-dim vectors
2. ChromaDB collection creation and management
3. Deterministic chunk IDs (re-crawling updates, not duplicates)
4. Full ingestion pipeline (pages → chunks → embeddings → ChromaDB)
5. Semantic search returns relevant results
"""
import shutil
import logging
from embeddings import (
    get_embedding_model,
    get_chroma_client,
    get_or_create_collection,
    _domain_to_collection_name,
    _make_chunk_id,
    ingest_pages,
    delete_collection,
    list_collections,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_embedding_model():
    print("=" * 60)
    print("TEST 1: Embedding Model Loads & Produces 384-dim Vectors")
    print("=" * 60)

    model = get_embedding_model()
    # Encode a sample sentence
    embedding = model.encode("Asyncio is a Python library for concurrent code.")

    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    assert embedding.shape == (384,), f"Expected 384 dimensions, got {embedding.shape}"
    assert abs(sum(embedding)) > 0, "Embedding should not be all zeros"

    # Verify singleton: calling again should return the same object
    model2 = get_embedding_model()
    assert model is model2, "Model should be a singleton (same object)"
    print("✅ Embedding model test passed!\n")


def test_collection_naming():
    print("=" * 60)
    print("TEST 2: Domain → Collection Name Mapping")
    print("=" * 60)

    name1 = _domain_to_collection_name("docs.python.org")
    name2 = _domain_to_collection_name("docs.python.org")
    name3 = _domain_to_collection_name("fastapi.tiangolo.dev")

    print(f"docs.python.org    → '{name1}'")
    print(f"docs.python.org    → '{name2}' (same domain, should match)")
    print(f"fastapi.tiangolo.dev → '{name3}'")

    assert name1 == name2, "Same domain should produce same collection name"
    assert name1 != name3, "Different domains should produce different names"
    assert 3 <= len(name1) <= 63, f"Collection name length must be 3-63, got {len(name1)}"
    assert name1[0].isalnum(), "Collection name must start with alphanumeric"
    print("✅ Collection naming test passed!\n")


def test_chunk_id_determinism():
    print("=" * 60)
    print("TEST 3: Chunk ID Determinism (Idempotent Upserts)")
    print("=" * 60)

    id1 = _make_chunk_id("https://docs.python.org/3/tutorial/index.html", 0)
    id2 = _make_chunk_id("https://docs.python.org/3/tutorial/index.html", 0)
    id3 = _make_chunk_id("https://docs.python.org/3/tutorial/index.html", 1)
    id4 = _make_chunk_id("https://docs.python.org/3/library/asyncio.html", 0)

    print(f"Same URL, same index:  {id1} == {id2} → {id1 == id2}")
    print(f"Same URL, diff index:  {id1} != {id3} → {id1 != id3}")
    print(f"Diff URL, same index:  {id1} != {id4} → {id1 != id4}")

    assert id1 == id2, "Same URL + same chunk_index must produce same ID"
    assert id1 != id3, "Different chunk_index must produce different ID"
    assert id1 != id4, "Different URL must produce different ID"
    print("✅ Chunk ID determinism test passed!\n")


def test_full_ingestion_pipeline():
    print("=" * 60)
    print("TEST 4: Full Ingestion Pipeline")
    print("=" * 60)

    # Simulate crawled pages
    fake_pages = [
        {
            "url": "https://test-docs.example.com/intro",
            "domain": "test-docs.example.com",
            "title": "Introduction to Testing",
            "text": (
                "Testing is a critical part of software development. "
                "It ensures that code behaves as expected under various conditions.\n\n"
                "Unit tests verify individual functions. Integration tests check how "
                "components work together. End-to-end tests simulate real user workflows.\n\n"
                "Python provides several testing frameworks including pytest, unittest, "
                "and doctest. Each has its own strengths and use cases."
            ),
            "depth": 0,
        },
        {
            "url": "https://test-docs.example.com/advanced",
            "domain": "test-docs.example.com",
            "title": "Advanced Testing Patterns",
            "text": (
                "Mocking is a technique where you replace real objects with simulated ones. "
                "This is useful when testing code that depends on external services.\n\n"
                "Fixtures in pytest provide a way to set up and tear down test resources. "
                "They can be scoped to function, class, module, or session level.\n\n"
                "Property-based testing with Hypothesis generates random test inputs "
                "to discover edge cases you might not think of manually."
            ),
            "depth": 1,
        },
    ]

    # Run ingestion
    result = ingest_pages(fake_pages)

    print(f"Domain: {result['domain']}")
    print(f"Pages ingested: {result['pages_ingested']}")
    print(f"Chunks stored: {result['chunks_stored']}")
    print(f"Collection: {result['collection_name']}")

    assert result["pages_ingested"] == 2
    assert result["chunks_stored"] > 0
    assert result["domain"] == "test-docs.example.com"

    # Verify data is actually in ChromaDB
    collection = get_or_create_collection("test-docs.example.com")
    count = collection.count()
    print(f"ChromaDB collection count: {count}")
    assert count == result["chunks_stored"], f"Expected {result['chunks_stored']} chunks in DB, got {count}"

    # Test semantic search: query should return relevant chunks
    model = get_embedding_model()
    query_embedding = model.encode("What is mocking in testing?").tolist()
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    print(f"\nSemantic search for 'What is mocking in testing?':")
    for i, (doc, meta) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
        print(f"  Result {i+1}: [{meta['title']}] {doc[:80]}...")

    # The top result should mention "mocking"
    top_doc = search_results["documents"][0][0].lower()
    assert "mock" in top_doc, f"Top result should mention 'mocking', got: {top_doc[:100]}"
    print("✅ Semantic search returned relevant results!")

    # Test idempotent upsert: re-ingesting should NOT increase count
    print(f"\nRe-ingesting same pages (idempotent upsert test)...")
    result2 = ingest_pages(fake_pages)
    count_after = collection.count()
    print(f"Count before re-ingest: {count}, Count after: {count_after}")
    assert count_after == count, f"Re-ingesting should NOT create duplicates! Before: {count}, After: {count_after}"
    print("✅ Idempotent upsert test passed!")

    # Cleanup: delete the test collection
    delete_collection("test-docs.example.com")
    print("✅ Full ingestion pipeline test passed!\n")


def main():
    print("\n🚀 Testing Phase 4: Embeddings + ChromaDB Storage...\n")
    test_embedding_model()
    test_collection_naming()
    test_chunk_id_determinism()
    test_full_ingestion_pipeline()
    print("=" * 60)
    print("🎉 ALL PHASE 4 TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
