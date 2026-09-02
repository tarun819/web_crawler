# Documentation Crawler and Hybrid RAG Search Engine

This project is a lightweight, local web crawler that builds a Retrieval-Augmented Generation (RAG) search engine for any public documentation website. You can give it a URL, and it will crawl the pages, chunk the text, generate vector embeddings, and allow you to ask natural language questions about the documentation.

## Overview

The system is built to run efficiently even in memory-constrained environments like cloud free tiers (e.g., Render with 512MB RAM). It avoids heavy frameworks like PyTorch, instead relying on ONNX Runtime for local embeddings, and uses a hybrid search approach to ensure accurate retrieval.

## How It Works

The system follows a simple pipeline:

1. A user provides a public documentation URL.
2. The crawler validates the URL and checks robots.txt before crawling.
3. Pages are discovered using an asynchronous BFS crawler.
4. Trafilatura extracts the main content from each HTML page.
5. The extracted text is split into token-limited chunks.
6. BGE-small generates an embedding for each chunk and stores it in ChromaDB.
7. When a user searches, the system performs both vector search and BM25 keyword search.
8. Reciprocal Rank Fusion combines the two result lists.
9. For AI questions, the retrieved chunks are passed to Groq, which generates an answer using only that context.

## Core Components

1. Crawler: An asynchronous BFS (Breadth-First Search) crawler that respects robots.txt and rate limits. It skips overly large files and handles timeouts gracefully.
2. Content Extraction: Uses Trafilatura to extract clean main-content text from HTML, stripping out navbars, footers, and boilerplate.
3. Embedding and Storage: Uses FastEmbed (ONNX) to generate BGE-small embeddings entirely locally, without making API calls to Hugging Face. The embeddings and chunks are stored in a local ChromaDB instance.
4. Hybrid Retrieval: Combines vector similarity search (cosine distance) with keyword search (BM25). The results are merged using Reciprocal Rank Fusion (RRF) to get the most relevant chunks.
5. Generation: Connects to the Groq API to generate an answer based only on the retrieved context, ensuring the LLM does not hallucinate information outside the documentation.

## Design Decisions

### Why no Redis?

The crawler runs as a single in-process worker and uses an asyncio.Queue and an in-memory set to keep track of crawl state. Since there are no multiple crawler processes competing over the same state, Redis would add infrastructure without solving a problem the current application has.

Redis would make more sense if the crawler needed to scale across multiple processes or machines.

### Why Trafilatura?

A documentation page contains a lot of content that is not actually part of the documentation, such as navigation menus, footers, and other boilerplate. Trafilatura is used to extract the main page content instead of relying on a hand-written HTML extraction heuristic.

BeautifulSoup/lxml are still used for discovering links.

### Why token-based chunking?

The embedding model has a token limit, so chunks are sized using the model's tokenizer rather than a simple word count. This prevents chunks from silently exceeding the model's input limit.

### Why hybrid search?

Vector search is good at understanding semantic similarity, while BM25 is particularly useful for exact technical terms such as configuration flags, function names, and error messages.

The two retrieval methods are combined using Reciprocal Rank Fusion because their raw scores are not directly comparable. RRF only needs the rank of each result rather than trying to normalize two different score scales.

## Requirements

Python 3.10 or higher is recommended.
You will need an active internet connection to crawl sites and a Groq API key to generate answers.

## Setup

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

3. Install the dependencies:
pip install -r requirements.txt

4. Configure your environment variables. Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=your_api_key_here

## Usage

You can start the server and the web interface using Uvicorn:

uvicorn main:app --port 8000

Once the server starts, open your web browser and go to http://localhost:8000 to access the Gradio user interface. 

From the interface, you can:
- Go to the Crawl tab, enter a documentation URL, and start crawling.
- Go to the Search Docs tab to run hybrid search queries.
- Go to the Ask AI tab to ask questions and get grounded answers with source citations.

## Security and Limits

Because the application accepts user-provided URLs, the crawler validates the target before fetching it.

It rejects non-HTTP(S) URLs and blocks private, loopback, and link-local IP addresses to reduce SSRF risks.

The crawler also respects robots.txt, applies crawl delays, limits response sizes, and uses per-page and overall crawl timeouts.

Crawl requests are rate-limited per IP to prevent a single user from repeatedly triggering expensive crawls.

## Retrieval Evaluation

The retrieval system was evaluated using a small set of manually written documentation queries.

The evaluation compares vector-only retrieval against the hybrid BM25 + vector approach using Hit@3 and MRR.

```
Vector-only      Hit@3: 100.0%    MRR: 1.0000
BM25-only        Hit@3: 100.0%    MRR: 0.8125
Hybrid (RRF)     Hit@3: 100.0%    MRR: 1.0000
```

The evaluation is intentionally small and is mainly used to verify that hybrid retrieval improves or maintains retrieval quality on technical documentation queries.

## Deployment

The application is deployed as a single FastAPI + Gradio service on Render.

The embedding model runs locally and Groq is used for answer generation, so the deployed application does not require a separate model server.

ChromaDB uses local storage. Render's free tier has ephemeral disk storage, so the stored documentation can be lost after a cold start or redeployment. This is acceptable for this project because crawling is on-demand and the documentation can simply be crawled again.

Live demo: https://web-crawler-0ao4.onrender.com/

## Project Structure

The main components are separated by responsibility:

- `crawler.py` handles crawling and page discovery.
- `content_extractor.py` handles HTML content and link extraction.
- `chunker.py` handles token-based text chunking.
- `embeddings.py` handles BGE embeddings and ChromaDB storage.
- `retrieval.py` handles vector search, BM25, and RRF.
- `rag.py` handles context-based answer generation with Groq.
- `main.py` exposes the FastAPI endpoints.
- `app.py` contains the Gradio interface.
