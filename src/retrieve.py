"""Step 4: dense search + BM25 search + Reciprocal Rank Fusion."""
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config

STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "on", "for", "it", "what", "does", "do", "how"}


def tokenize(text: str) -> list[str]:
    """Same tokenizer for indexing AND querying (a classic bug source if they differ)."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def rrf(rankings: list[list[int]], k: int = config.RRF_K) -> list[int]:
    """Reciprocal Rank Fusion. Each ranking is a list of chunk indices, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self):
        chunks_path = config.INDEX_DIR / "chunks.json"
        if not chunks_path.exists():
            raise SystemExit("Index not found. Run:  python -m src.index")
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(config.INDEX_DIR / "embeddings.npy")
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def dense_search(self, query: str, k: int) -> list[int]:
        q = self.model.encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.embeddings @ q
        return [int(i) for i in np.argsort(-scores)[:k]]

    def bm25_search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]

    def hybrid_search(self, query: str) -> list[int]:
        dense = self.dense_search(query, config.TOP_K_DENSE)
        sparse = self.bm25_search(query, config.TOP_K_BM25)
        return rrf([dense, sparse])
