# Documentation Crawler and Hybrid RAG Search Engine

This project is a lightweight, local web crawler that builds a Retrieval-Augmented Generation (RAG) search engine for any public documentation website. You can give it a URL, and it will crawl the pages, chunk the text, generate vector embeddings, and allow you to ask natural language questions about the documentation.

## Overview

The system is built to run efficiently even in memory-constrained environments like cloud free tiers (e.g., Render with 512MB RAM). It avoids heavy frameworks like PyTorch, instead relying on ONNX Runtime for local embeddings, and uses a hybrid search approach to ensure accurate retrieval.

## Core Components

1. Crawler
An asynchronous BFS (Breadth-First Search) crawler that respects robots.txt and rate limits. It skips overly large files and handles timeouts gracefully.

2. Content Extraction
Uses Trafilatura to extract clean main-content text from HTML, stripping out navbars, footers, and boilerplate.

3. Embedding and Storage
Uses FastEmbed (ONNX) to generate BGE-small embeddings entirely locally, without making API calls to Hugging Face. The embeddings and chunks are stored in a local ChromaDB instance.

4. Hybrid Retrieval
Combines vector similarity search (cosine distance) with keyword search (BM25). The results are merged using Reciprocal Rank Fusion (RRF) to get the most relevant chunks.

5. Generation
Connects to the Groq API to generate an answer based only on the retrieved context, ensuring the LLM does not hallucinate information outside the documentation.

## Requirements

Python 3.10 or higher is recommended.
You will need an active internet connection to crawl sites and a Groq API key to generate answers.

## Setup Instructions

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

## Deployment Notes

If you are deploying this to a cloud environment like Render, the crawler is already optimized to run on 1 thread with small batch sizes to prevent memory overflow on 512MB instances.
