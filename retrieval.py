"""
Hybrid retrieval: Vector search (ChromaDB) + BM25 keyword search + RRF fusion.

Why hybrid?
- Vector search finds semantically similar chunks (synonyms, paraphrasing).
- BM25 finds exact keyword matches (class names, error codes, identifiers).
- Neither alone is sufficient; combining them via RRF gives best-of-both.

Reciprocal Rank Fusion (RRF):
  Fuses two ranked lists by rank position (not raw scores), avoiding the
  incompatible-scales problem (cosine: 0-1 vs BM25: 0-25+).
  Formula: RRF_score = 1/(k + rank_vector) + 1/(k + rank_bm25)
"""
import logging
import time
from typing import Optional

from rank_bm25 import BM25Okapi

import config
from embeddings import get_embedding_model, get_or_create_collection

logger = logging.getLogger(__name__)

# In-memory BM25 index cache (one per domain, rebuilt when chunks change)
_bm25_cache: dict[str, dict] = {}


# =====================================================================
# BM25 Keyword Index
# =====================================================================

def _build_bm25_index(domain: str) -> dict:
    """
    Build a BM25 index from all chunks in a domain's ChromaDB collection.

    Fetches all documents + metadata from ChromaDB, tokenizes them into
    word lists, and creates an in-memory BM25Okapi index for keyword search.

    The index is cached per domain and invalidated when chunk count changes.
    """
    collection = get_or_create_collection(domain)
    count = collection.count()

    # Return cached index if chunk count hasn't changed
    if domain in _bm25_cache and _bm25_cache[domain]["count"] == count:
        return _bm25_cache[domain]

    if count == 0:
        _bm25_cache[domain] = {"count": 0, "index": None, "docs": [], "metadatas": [], "ids": []}
        return _bm25_cache[domain]

    # Fetch all documents from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    # Tokenize documents into word lists for BM25
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25_index = BM25Okapi(tokenized_docs)

    _bm25_cache[domain] = {
        "count": count,
        "index": bm25_index,
        "docs": docs,
        "metadatas": metadatas,
        "ids": ids,
    }

    logger.info(f"BM25 index built for domain '{domain}' with {count} chunks.")
    return _bm25_cache[domain]


def invalidate_bm25_cache(domain: str) -> None:
    """Invalidate BM25 cache for a domain (call after new crawl/ingestion)."""
    if domain in _bm25_cache:
        del _bm25_cache[domain]
        logger.info(f"BM25 cache invalidated for domain '{domain}'.")


# =====================================================================
# Individual Search Methods
# =====================================================================

def vector_search(query: str, domain: str, top_k: int = config.TOP_K_VECTOR) -> list[dict]:
    """
    Semantic vector search via ChromaDB.

    Embeds the query using BGE-small, then finds the top-k closest chunks
    by cosine similarity in the domain's collection.
    """
    model = get_embedding_model()
    collection = get_or_create_collection(domain)

    if collection.count() == 0:
        return []

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ranked = []
    for i in range(len(results["ids"][0])):
        ranked.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return ranked


def bm25_search(query: str, domain: str, top_k: int = config.TOP_K_BM25) -> list[dict]:
    """
    BM25 keyword search over all chunks in a domain.

    Tokenizes the query into words and scores every chunk using
    term frequency and inverse document frequency.
    """
    cache = _build_bm25_index(domain)

    if cache["index"] is None:
        return []

    query_tokens = query.lower().split()
    scores = cache["index"].get_scores(query_tokens)

    # Pair scores with document data and sort descending
    scored_docs = []
    for i, score in enumerate(scores):
        if score > 0:
            scored_docs.append({
                "id": cache["ids"][i],
                "text": cache["docs"][i],
                "metadata": cache["metadatas"][i],
                "bm25_score": float(score),
            })

    scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_docs[:top_k]


# =====================================================================
# Reciprocal Rank Fusion (RRF)
# =====================================================================

def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = config.RRF_K,
    top_n: int = config.TOP_K_RESULTS,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF uses rank positions instead of raw scores, which avoids the
    incompatible-scales problem between cosine similarity and BM25.

    Formula: RRF_score(doc) = 1/(k + rank_in_vector) + 1/(k + rank_in_bm25)

    Documents appearing in only one list still get a score from that list
    (their rank in the missing list is treated as infinity, contributing 0).
    """
    # Build a map: chunk_id → {data, rrf_score}
    fused: dict[str, dict] = {}

    # Score from vector search rankings
    for rank, result in enumerate(vector_results):
        chunk_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)  # rank+1 because ranks are 1-indexed in RRF
        fused[chunk_id] = {
            "id": chunk_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "rrf_score": rrf_score,
            "vector_rank": rank + 1,
            "bm25_rank": None,
        }

    # Add scores from BM25 rankings
    for rank, result in enumerate(bm25_results):
        chunk_id = result["id"]
        rrf_score = 1.0 / (k + rank + 1)

        if chunk_id in fused:
            # Chunk appeared in both lists — add BM25 RRF score
            fused[chunk_id]["rrf_score"] += rrf_score
            fused[chunk_id]["bm25_rank"] = rank + 1
        else:
            # Chunk appeared only in BM25
            fused[chunk_id] = {
                "id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "rrf_score": rrf_score,
                "vector_rank": None,
                "bm25_rank": rank + 1,
            }

    # Sort by fused RRF score (highest first) and return top_n
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_n]


# =====================================================================
# Main Hybrid Search (Public API)
# =====================================================================

def hybrid_search(query: str, domain: str) -> dict:
    """
    Run hybrid retrieval: vector + BM25 + RRF fusion.

    Returns the top-k fused results along with latency breakdown.
    """
    t_start = time.perf_counter()

    # Vector search
    t_vec = time.perf_counter()
    vec_results = vector_search(query, domain)
    vector_ms = (time.perf_counter() - t_vec) * 1000

    # BM25 search
    t_bm25 = time.perf_counter()
    bm25_results = bm25_search(query, domain)
    bm25_ms = (time.perf_counter() - t_bm25) * 1000

    # RRF fusion
    t_fuse = time.perf_counter()
    fused = reciprocal_rank_fusion(vec_results, bm25_results)
    fusion_ms = (time.perf_counter() - t_fuse) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"Hybrid search for '{query[:50]}...' on '{domain}': "
        f"vector={vector_ms:.1f}ms, bm25={bm25_ms:.1f}ms, fusion={fusion_ms:.1f}ms, total={total_ms:.1f}ms"
    )

    return {
        "query": query,
        "domain": domain,
        "results": fused,
        "latency": {
            "vector_ms": round(vector_ms, 1),
            "bm25_ms": round(bm25_ms, 1),
            "fusion_ms": round(fusion_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }
