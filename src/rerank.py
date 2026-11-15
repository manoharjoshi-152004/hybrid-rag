"""Step 5: re-score (query, chunk) pairs with a cross-encoder."""
from sentence_transformers import CrossEncoder

from src import config


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(config.RERANK_MODEL)

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not chunks:
            return []
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])[:top_n]
        return [{**chunk, "score": float(score)} for score, chunk in ranked]
