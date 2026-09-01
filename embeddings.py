"""
Embedding generation and ChromaDB storage.

Responsibilities:
- Load the BGE-small embedding model once (singleton pattern).
- Manage per-domain ChromaDB collections.
- Generate deterministic chunk IDs for idempotent upserts (re-crawling
  updates existing chunks instead of creating duplicates).
- Ingest crawled + chunked pages into ChromaDB with source metadata.

No network calls - all local compute and local disk storage.
"""
import hashlib
import logging
import re
from typing import Optional, Union, List

import chromadb
from fastembed import TextEmbedding
import numpy as np

import config
from chunker import chunk_text, get_tokenizer

logger = logging.getLogger(__name__)


class FastEmbedWrapper:
    """
    Drop-in wrapper around fastembed.TextEmbedding that provides
    an .encode() method matching SentenceTransformer's interface.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self._model = TextEmbedding(model_name=model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Embed a single text string or a list of text strings using fastembed.

        Returns:
            - 1D numpy array shape (384,) if texts is a single str (calling .tolist() returns list[float])
            - 2D numpy array shape (N, 384) if texts is a list[str] (calling .tolist() returns list[list[float]])
        """
        if isinstance(texts, str):
            embeddings = list(self._model.embed([texts], batch_size=1))
            return embeddings[0]
        else:
            embeddings = list(self._model.embed(texts, batch_size=batch_size))
            return np.array(embeddings)

    @property
    def tokenizer(self):
        """Access the local ONNX fast tokenizer bundled with the FastEmbed model."""
        return self._model.model.tokenizer


_model: Optional[FastEmbedWrapper] = None
_chroma_client: Optional[chromadb.ClientAPI] = None


def get_embedding_model() -> FastEmbedWrapper:
    """
    Load the BGE-small ONNX embedding model via FastEmbed once and cache it.
    """
    global _model
    if _model is None:
        logger.info(f"Loading FastEmbed model: {config.EMBEDDING_MODEL}")
        _model = FastEmbedWrapper(config.EMBEDDING_MODEL)
        logger.info("FastEmbed model loaded successfully.")
    return _model


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Initialize a persistent ChromaDB client once and cache it.

    Data is stored locally on disk at CHROMA_PERSIST_DIR (./chroma_data/).
    On Render free tier, this resets on cold start - which is fine because
    crawling is on-demand and re-crawling re-populates the collection.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at: {config.CHROMA_PERSIST_DIR}")
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        logger.info("ChromaDB client initialized.")
    return _chroma_client


# =====================================================================
# Collection Management (one collection per domain)
# =====================================================================

def _domain_to_collection_name(domain: str) -> str:
    """
    Convert a domain string into a valid ChromaDB collection name.

    ChromaDB collection names must:
    - Be 3-63 characters long
    - Start and end with an alphanumeric character
    - Contain only alphanumerics, underscores, or hyphens
    - Not contain consecutive periods (..)

    We use: "domain_" + md5 hash prefix (first 12 chars) for uniqueness,
    plus a sanitized domain prefix for human readability.
    """
    # Sanitize domain: replace dots and non-alphanumerics with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', domain.lower())
    # Truncate to keep total name under 63 chars
    sanitized = sanitized[:30]
    # Add hash suffix for uniqueness (two domains could sanitize to the same string)
    domain_hash = hashlib.md5(domain.encode()).hexdigest()[:12]
    name = f"{sanitized}_{domain_hash}"
    # Ensure it starts and ends with alphanumeric
    name = name.strip("_")
    return name


def get_or_create_collection(domain: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection for the given domain."""
    client = get_chroma_client()
    collection_name = _domain_to_collection_name(domain)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"domain": domain, "hnsw:space": "cosine"},
    )
    logger.info(f"Collection '{collection_name}' ready (domain: {domain}, count: {collection.count()}).")
    return collection


def list_collections() -> list[str]:
    """List all domain collections in ChromaDB."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def delete_collection(domain: str) -> None:
    """Delete a domain collection from ChromaDB."""
    client = get_chroma_client()
    collection_name = _domain_to_collection_name(domain)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except ValueError:
        logger.warning(f"Collection '{collection_name}' not found for deletion.")


# =====================================================================
# Chunk ID Generation (deterministic for idempotent upserts)
# =====================================================================

def _make_chunk_id(url: str, chunk_index: int) -> str:
    """
    Generate a deterministic, unique ID for each chunk.

    Uses md5(url + "::" + chunk_index) so that:
    - Re-crawling the same URL produces the same IDs -> upsert (update, not duplicate).
    - Different URLs or chunk indices always produce different IDs.
    """
    raw = f"{url}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# =====================================================================
# Ingestion Pipeline: Crawled Pages -> Chunks -> Embeddings -> ChromaDB
# =====================================================================

def ingest_pages(pages: list[dict]) -> dict:
    """
    Full ingestion pipeline: takes crawled page dicts, chunks them,
    embeds them, and upserts into ChromaDB.

    Args:
        pages: List of crawled page dicts from crawler.crawl().
               Each dict has: {"url", "domain", "title", "text", "depth"}

    Returns:
        Summary dict: {"domain", "pages_ingested", "chunks_stored", "collection_name"}
    """
    if not pages:
        return {"domain": "", "pages_ingested": 0, "chunks_stored": 0, "collection_name": ""}

    domain = pages[0]["domain"]
    model = get_embedding_model()
    tokenizer = get_tokenizer()
    collection = get_or_create_collection(domain)

    all_ids = []
    all_embeddings = []
    all_documents = []
    all_metadatas = []

    for page in pages:
        # Chunk the page text
        chunks = chunk_text(page["text"], tokenizer=tokenizer)

        for chunk in chunks:
            chunk_id = _make_chunk_id(page["url"], chunk["chunk_index"])
            chunk_text_str = chunk["text"]

            all_ids.append(chunk_id)
            all_documents.append(chunk_text_str)
            all_metadatas.append({
                "url": page["url"],
                "title": page["title"],
                "domain": domain,
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
            })

    if not all_documents:
        logger.warning("No chunks produced from pages. Nothing to ingest.")
        return {"domain": domain, "pages_ingested": len(pages), "chunks_stored": 0,
                "collection_name": _domain_to_collection_name(domain)}

    # Batch embed all chunks at once (much faster than one-by-one)
    logger.info(f"Embedding {len(all_documents)} chunks for domain '{domain}'...")
    all_embeddings = model.encode(all_documents, batch_size=32, show_progress_bar=False).tolist()

    # Upsert into ChromaDB (idempotent: same IDs update, not duplicate)
    # ChromaDB has a batch size limit, so we upsert in batches of 500
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        end = min(i + batch_size, len(all_ids))
        collection.upsert(
            ids=all_ids[i:end],
            embeddings=all_embeddings[i:end],
            documents=all_documents[i:end],
            metadatas=all_metadatas[i:end],
        )

    total_chunks = len(all_ids)
    collection_name = _domain_to_collection_name(domain)
    total_in_coll = collection.count()
    logger.info(
        f"Ingestion complete: {len(pages)} pages -> {total_chunks} chunks "
        f"into collection {collection_name} (total in collection: {total_in_coll})."
    )

    return {
        "domain": domain,
        "pages_ingested": len(pages),
        "chunks_stored": total_chunks,
        "collection_name": collection_name,
    }
