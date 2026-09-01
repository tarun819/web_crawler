"""
Gradio Web UI for the Documentation Crawler & RAG Search Engine.

Three-tab interface:
  Tab 1 (Crawl): Enter a URL → crawl, chunk, embed into ChromaDB.
  Tab 2 (Search): Hybrid search (Vector + BM25 + RRF) → raw ranked chunks.
  Tab 3 (Ask AI): Grounded RAG → synthesized answer with source citations.
"""
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file (GROQ_API_KEY) before any module imports

import asyncio
import logging

import gradio as gr

from crawler import crawl
from embeddings import ingest_pages, list_collections, get_or_create_collection
from retrieval import hybrid_search, invalidate_bm25_cache
from rag import query_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Helper: get list of crawled domains for the dropdown
def get_crawled_domains() -> list[str]:
    """Extract domain names from ChromaDB collection metadata."""
    client_collections = list_collections()
    if not client_collections:
        return []

    from embeddings import get_chroma_client
    client = get_chroma_client()
    domains = []
    for col in client.list_collections():
        meta = col.metadata or {}
        domain = meta.get("domain", col.name)
        domains.append(domain)
    return sorted(set(domains))


# Tab 1: Crawl Handler
async def handle_crawl(url: str, max_pages: int, max_depth: int, progress=gr.Progress()):
    """Crawl a documentation site and ingest into ChromaDB."""
    if not url or not url.strip():
        return "⚠️ Please enter a valid URL.", gr.update(choices=get_crawled_domains())

    url = url.strip()
    progress(0.1, desc="Starting crawl...")

    try:
        # Phase 2: Crawl
        progress(0.2, desc=f"Crawling {url}...")
        pages = await crawl(
            seed_url=url,
            max_pages=int(max_pages),
            max_depth=int(max_depth),
        )

        if not pages:
            return (
                "❌ **Crawl returned no pages.**\n\n"
                "Possible reasons:\n"
                "- The URL is unreachable or blocked by robots.txt\n"
                "- The URL points to a private/local address (SSRF blocked)\n"
                "- The site returned no extractable text content",
                gr.update(choices=get_crawled_domains()),
            )

        # Phase 3+4: Chunk + Embed + Store
        progress(0.6, desc=f"Chunking and embedding {len(pages)} pages...")
        result = ingest_pages(pages)

        # Invalidate BM25 cache for fresh searches
        invalidate_bm25_cache(result["domain"])

        progress(1.0, desc="Done!")

        status = (
            f"✅ **Crawl Complete!**\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| **Domain** | `{result['domain']}` |\n"
            f"| **Pages Crawled** | {result['pages_ingested']} |\n"
            f"| **Chunks Stored** | {result['chunks_stored']} |\n"
            f"| **Collection** | `{result['collection_name']}` |\n\n"
            f"You can now search or ask questions in the other tabs!"
        )
        return status, gr.update(choices=get_crawled_domains(), value=result["domain"])

    except Exception as e:
        logger.error(f"Crawl error: {e}", exc_info=True)
        return f"❌ **Error:** {str(e)}", gr.update(choices=get_crawled_domains())


# Tab 2: Hybrid Search Handler
def handle_search(query: str, domain: str):
    """Run hybrid search and return formatted results."""
    if not query or not query.strip():
        return "⚠️ Please enter a search query."
    if not domain:
        return "⚠️ Please select a crawled domain first."

    collection = get_or_create_collection(domain)
    if collection.count() == 0:
        return f"⚠️ No data found for domain `{domain}`. Please crawl it first."

    result = hybrid_search(query.strip(), domain)
    chunks = result.get("results", [])
    latency = result.get("latency", {})

    if not chunks:
        return "No results found for your query."

    # Format results as clean markdown
    output_parts = []
    output_parts.append(
        f"**Found {len(chunks)} results** "
        f"(Vector: {latency.get('vector_ms', 0)}ms | "
        f"BM25: {latency.get('bm25_ms', 0)}ms | "
        f"Total: {latency.get('total_ms', 0)}ms)\n\n---\n"
    )

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Untitled")
        url = meta.get("url", "")
        rrf = chunk.get("rrf_score", 0)
        vec_rank = chunk.get("vector_rank", "—")
        bm25_rank = chunk.get("bm25_rank", "—")
        text = chunk.get("text", "")

        # Truncate long chunks for display
        display_text = text[:300] + "..." if len(text) > 300 else text

        output_parts.append(
            f"### Result {i}: [{title}]({url})\n"
            f"**RRF Score:** `{rrf:.4f}` | "
            f"**Vector Rank:** `{vec_rank}` | "
            f"**BM25 Rank:** `{bm25_rank}`\n\n"
            f"> {display_text}\n\n---\n"
        )

    return "\n".join(output_parts)


# Tab 3: RAG Query Handler
def handle_query(query: str, domain: str):
    """Run grounded RAG and return formatted answer with citations."""
    if not query or not query.strip():
        return "⚠️ Please enter a question.", ""
    if not domain:
        return "⚠️ Please select a crawled domain first.", ""

    collection = get_or_create_collection(domain)
    if collection.count() == 0:
        return f"⚠️ No data found for domain `{domain}`. Please crawl it first.", ""

    try:
        result = query_rag(query.strip(), domain)

        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        latency = result.get("latency", {})
        chunks_used = result.get("chunks_used", 0)

        # Format source citations
        source_parts = []
        if sources:
            source_parts.append("### 📚 Sources\n")
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                source_parts.append(f"{i}. [{title}]({url})")

        source_parts.append(
            f"\n\n---\n"
            f"⚡ **Latency:** Retrieval: {latency.get('retrieval_ms', 0)}ms | "
            f"LLM: {latency.get('llm_ms', 0)}ms | "
            f"Total: {latency.get('total_ms', 0)}ms | "
            f"Chunks used: {chunks_used}"
        )

        return answer, "\n".join(source_parts)

    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        return f"❌ **Error generating answer:** {str(e)}", ""


# Build the Gradio UI
with gr.Blocks(
    title="Documentation RAG Search Engine",
) as demo:

    gr.Markdown(
        "# 🌐 Documentation Crawler & RAG Search Engine\n"
        "Crawl any public documentation site, then search or ask questions about it."
    )

    # Shared domain selector (updated after crawling)
    with gr.Row():
        domain_dropdown = gr.Dropdown(
            choices=get_crawled_domains(),
            label="📂 Select Crawled Domain",
            interactive=True,
            scale=3,
        )
        refresh_btn = gr.Button("🔄 Refresh", scale=1)

    refresh_btn.click(
        fn=lambda: gr.update(choices=get_crawled_domains()),
        outputs=domain_dropdown,
    )

    # Tab 1: Crawl
    with gr.Tab("🕷️ Crawl"):
        gr.Markdown("Enter a public documentation URL to crawl, chunk, and index.")

        with gr.Row():
            crawl_url = gr.Textbox(
                label="Documentation URL",
                placeholder="https://docs.python.org/3/library/asyncio.html",
                scale=3,
            )
            crawl_pages = gr.Slider(
                minimum=1, maximum=50, value=10, step=1,
                label="Max Pages", scale=1,
            )
            crawl_depth = gr.Slider(
                minimum=0, maximum=5, value=2, step=1,
                label="Max Depth", scale=1,
            )

        crawl_btn = gr.Button("🚀 Start Crawling", variant="primary")
        crawl_output = gr.Markdown(label="Crawl Status")

        crawl_btn.click(
            fn=handle_crawl,
            inputs=[crawl_url, crawl_pages, crawl_depth],
            outputs=[crawl_output, domain_dropdown],
        )

    # Tab 2: Hybrid Search
    with gr.Tab("🔍 Search Docs"):
        gr.Markdown("Search crawled documentation using Hybrid Retrieval (Vector + BM25 + RRF).")

        search_query = gr.Textbox(
            label="Search Query",
            placeholder="How to handle HTTP timeouts?",
        )
        search_btn = gr.Button("🔍 Search", variant="primary")
        search_output = gr.Markdown(label="Search Results")

        search_btn.click(
            fn=handle_search,
            inputs=[search_query, domain_dropdown],
            outputs=search_output,
        )

    # Tab 3: Ask AI (RAG)
    with gr.Tab("🤖 Ask AI"):
        gr.Markdown("Ask a question and get a grounded answer with source citations from crawled documentation.")

        query_input = gr.Textbox(
            label="Your Question",
            placeholder="What is asyncio.gather() and how do I use it?",
        )
        query_btn = gr.Button("🤖 Ask Question", variant="primary")

        answer_output = gr.Markdown(label="Answer")
        sources_output = gr.Markdown(label="Sources & Latency")

        query_btn.click(
            fn=handle_query,
            inputs=[query_input, domain_dropdown],
            outputs=[answer_output, sources_output],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
