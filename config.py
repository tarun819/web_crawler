import os

# --- Crawler Settings ---
MAX_PAGES = 50
MAX_DEPTH = 3
CRAWL_DELAY = 0.75           # Seconds between requests to the same domain
REQUEST_TIMEOUT = 10.0       # Per-request timeout in seconds
MAX_RETRIES = 3              # Max retry attempts for 429/5xx status codes
MAX_RESPONSE_SIZE = 2 * 1024 * 1024   # 2MB — skip oversized pages before parsing
CRAWL_SESSION_TIMEOUT = 150           # 2.5 min hard deadline for entire crawl session

# --- Embedding & Storage ---
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./chroma_data"

# --- Chunking Settings ---
# Measured in TOKENS (using the model's actual tokenizer, not word count).
# BGE-small's hard limit is 512 tokens — target 400 to stay safely within it.
CHUNK_SIZE = 400             # Target tokens per chunk
CHUNK_OVERLAP = 100          # Overlap tokens between adjacent chunks

# --- Retrieval & Fusion ---
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_RESULTS = 5
RRF_K = 60                   # Standard Reciprocal Rank Fusion constant

# --- LLM Settings ---
GROQ_MODEL = "llama-3.2-3b-preview"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
