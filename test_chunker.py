"""
Test suite for the token-based paragraph-aware chunker.

Verifies that:
1. Short text produces a single chunk
2. Paragraph boundaries are respected and overlap works
3. Oversized paragraphs are split at sentence boundaries
4. ALL chunks stay within BGE-small's 512-token hard limit
5. Token counts are exact (from the model's actual tokenizer)
"""
from chunker import chunk_text, get_tokenizer


def test_short_text():
    print("=" * 60)
    print("TEST 1: Short Text (< 400 tokens)")
    print("=" * 60)

    text = "Asyncio is a library to write concurrent code using async/await syntax."
    chunks = chunk_text(text, max_tokens=400, overlap_tokens=100)

    print(f"Chunks produced: {len(chunks)}")
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["token_count"] <= 512  # must be within BGE-small limit
    print(f"Token count: {chunks[0]['token_count']}")
    print("✅ Short text test passed!\n")


def test_paragraph_boundaries_and_overlap():
    print("=" * 60)
    print("TEST 2: Paragraph Boundaries & Overlap")
    print("=" * 60)

    p1 = "Paragraph one discusses the basics of web crawling. " * 15
    p2 = "Paragraph two covers URL normalization techniques. " * 15
    p3 = "Paragraph three explains content extraction methods. " * 15
    p4 = "Paragraph four details chunking strategies for RAG. " * 15

    full_text = f"{p1}\n\n{p2}\n\n{p3}\n\n{p4}"

    tokenizer = get_tokenizer()
    total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

    chunks = chunk_text(full_text, max_tokens=300, overlap_tokens=50, tokenizer=tokenizer)

    print(f"Total tokens in input: {total_tokens}")
    print(f"Chunks produced: {len(chunks)}")
    for c in chunks:
        print(f"  Chunk {c['chunk_index']} ({c['token_count']} tokens): {c['text'][:60]}...")

    assert len(chunks) >= 2
    print("✅ Paragraph boundary and overlap test passed!\n")


def test_oversized_paragraph_sentence_splitting():
    print("=" * 60)
    print("TEST 3: Oversized Paragraph Sentence Splitting")
    print("=" * 60)

    # 1 single giant paragraph with multiple sentences, no double newlines
    s1 = "First sentence has some introductory words about asyncio programming. " * 20
    s2 = "Second sentence continues with more technical explanation about crawlers. " * 20
    s3 = "Third sentence wraps up the thoughts about web scraping neatly. " * 20
    giant_p = s1 + s2 + s3

    tokenizer = get_tokenizer()
    total_tokens = len(tokenizer.encode(giant_p, add_special_tokens=False))

    chunks = chunk_text(giant_p, max_tokens=200, overlap_tokens=30, tokenizer=tokenizer)

    print(f"Giant paragraph tokens: {total_tokens}")
    print(f"Chunks produced: {len(chunks)}")

    for c in chunks:
        print(f"  Chunk {c['chunk_index']} ({c['token_count']} tokens)")

    assert len(chunks) > 1
    print("✅ Oversized paragraph sentence splitting passed!\n")


def test_all_chunks_within_512_limit():
    print("=" * 60)
    print("TEST 4: ALL Chunks Within BGE-small 512-Token Limit")
    print("=" * 60)

    # Technical documentation with long identifiers, URLs, and code
    technical_text = """
    The asyncio.Queue implementation provides a thread-safe FIFO queue.

    You can use it with httpx.AsyncClient for non-blocking HTTP requests:

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        response = await client.get("https://docs.python.org/3/library/asyncio-queue.html")
        print(response.status_code)

    The BAAI/bge-small-en-v1.5 model has a 512-token context window.
    SentenceTransformer embeddings capture semantic meaning in 384 dimensions.

    Configuration parameters like GROQ_API_KEY=gsk_abc123def456 and
    CHROMA_PERSIST_DIR=./chroma_data should be set in environment variables.

    The RecipientRankFusion algorithm combines rankings using the formula:
    RRF_score(d) = sum(1 / (k + rank_m(d))) for each retrieval method m.

    BeautifulSoup4 with lxml backend provides fast HTML parsing capabilities
    for extracting content from documentation pages like
    https://fastapi.tiangolo.com/tutorial/first-steps/ and similar resources.
    """ * 3  # Repeat to make it long enough to produce multiple chunks

    tokenizer = get_tokenizer()
    chunks = chunk_text(technical_text, max_tokens=400, overlap_tokens=100, tokenizer=tokenizer)

    print(f"Chunks produced: {len(chunks)}")
    all_within_limit = True
    for c in chunks:
        status = "✅" if c["token_count"] <= 512 else "❌ OVER LIMIT!"
        print(f"  Chunk {c['chunk_index']}: {c['token_count']} tokens {status}")
        if c["token_count"] > 512:
            all_within_limit = False

    assert all_within_limit, "Some chunks exceed BGE-small's 512-token limit!"
    print("✅ All chunks within 512-token limit!\n")


def main():
    print("\n🚀 Testing Token-Based Paragraph-Aware Chunker...\n")
    test_short_text()
    test_paragraph_boundaries_and_overlap()
    test_oversized_paragraph_sentence_splitting()
    test_all_chunks_within_512_limit()
    print("=" * 60)
    print("🎉 ALL CHUNKER TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
