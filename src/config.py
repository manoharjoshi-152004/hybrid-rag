"""All tunable settings live here so you can change one thing at a time."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index_store"

# --- Chunking ---
CHUNK_SIZE = 600       # characters
CHUNK_OVERLAP = 90     # characters (~15%)

# --- Models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models want this prefix on QUERIES only (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Retrieval funnel: wide -> narrow ---
TOP_K_DENSE = 30        # candidates from embeddings
TOP_K_BM25 = 30         # candidates from keywords
TOP_K_RERANK_INPUT = 20 # fused candidates sent to the reranker
TOP_K_FINAL = 4         # chunks sent to the LLM
RRF_K = 60
