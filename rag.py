"""
Grounded Answer Generation with Groq LLM.

Responsibilities:
- Takes top-k retrieved chunks from hybrid search (Phase 5).
- Stuffs chunks into a strictly grounded context prompt.
- Instructs Groq's LLaMA 3.2 3B model to answer ONLY from context with citations.
- Implements retry with exponential backoff for Groq API rate limits (HTTP 429).
- Tracks full latency breakdown (retrieval_ms, llm_ms, total_ms).
"""
import logging
import os
import time
from typing import Optional

from groq import Groq, RateLimitError, APIError

import config
from retrieval import hybrid_search

logger = logging.getLogger(__name__)

# Singleton Groq client
_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Initialize and cache the Groq API client."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", config.GROQ_API_KEY)
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please export GROQ_API_KEY='gsk_...' or set it in your environment."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# Prompt Formatting & Grounding

SYSTEM_PROMPT = """You are a precise technical documentation assistant.
Your goal is to answer the user's question using ONLY the provided documentation context below.

Rules:
1. Answer strictly based on the provided context. Do NOT use outside knowledge or make assumptions.
2. If the provided context does NOT contain enough information to answer the question, state:
   "I do not have enough information in the crawled documentation to answer this question."
3. Always cite your sources. When stating facts or instructions, refer to the source using [Source N] or the document title/URL.
4. Format code blocks with appropriate language tags (e.g., ```python).
5. Be concise, direct, and technically accurate."""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context blocks for the LLM."""
    if not chunks:
        return "No relevant documentation chunks found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Untitled")
        url = meta.get("url", "")
        text = chunk.get("text", "")

        context_parts.append(
            f"--- [Source {i}: {title}] ({url}) ---\n{text}"
        )

    return "\n\n".join(context_parts)


# LLM Generation with Retries

def _call_groq_with_retry(
    client: Groq,
    messages: list[dict],
    model: str = config.GROQ_MODEL,
    max_retries: int = 3,
) -> str:
    """Call Groq API with exponential backoff on rate limits (429)."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,  # Low temperature for factual, grounded answers
                max_tokens=800,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.warning(f"Groq 429 Rate Limit. Retrying in {backoff}s... ({e})")
                time.sleep(backoff)
            else:
                logger.error(f"Groq rate limit exceeded after {max_retries} attempts.")
                raise
        except APIError as e:
            logger.error(f"Groq API error: {e}")
            raise


# Main RAG Query Pipeline (Public API)

def query_rag(query: str, domain: str) -> dict:
    """
    Complete Grounded RAG Pipeline:
    1. Runs hybrid retrieval (Vector + BM25 + RRF) on domain chunks.
    2. Builds context-stuffed prompt.
    3. Calls Groq LLM for a citation-grounded response.
    4. Returns answer, source citations, and full latency metrics.
    """
    t_start = time.perf_counter()

    # Step 1: Hybrid Retrieval (Phase 5)
    search_result = hybrid_search(query, domain)
    chunks = search_result.get("results", [])
    retrieval_ms = search_result.get("latency", {}).get("total_ms", 0.0)

    # Step 2: Extract unique sources for UI citation list
    sources = []
    seen_urls = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        url = meta.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "title": meta.get("title", url),
                "url": url,
                "domain": meta.get("domain", domain),
            })

    # Step 3: Format Context & Prompt
    context_text = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Documentation Context:\n{context_text}\n\nUser Question: {query}",
        },
    ]

    # Step 4: Call Groq LLM
    t_llm = time.perf_counter()
    client = get_groq_client()
    answer = _call_groq_with_retry(client, messages)
    llm_ms = (time.perf_counter() - t_llm) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"RAG query complete for '{query[:40]}...': "
        f"retrieval={retrieval_ms:.1f}ms, llm={llm_ms:.1f}ms, total={total_ms:.1f}ms"
    )

    return {
        "query": query,
        "domain": domain,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "latency": {
            "retrieval_ms": round(retrieval_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }
