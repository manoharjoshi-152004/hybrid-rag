"""Glue: one function that runs retrieval in any of the 4 configurations."""
from src import config


def search(retriever, reranker, query: str, mode: str = "hybrid",
           use_rerank: bool = True, final_k: int = config.TOP_K_FINAL) -> list[dict]:
    """mode: 'dense' | 'bm25' | 'hybrid'. Returns a list of chunk dicts."""
    if mode == "dense":
        ids = retriever.dense_search(query, config.TOP_K_DENSE)
    elif mode == "bm25":
        ids = retriever.bm25_search(query, config.TOP_K_BM25)
    else:
        ids = retriever.hybrid_search(query)

    candidates = [retriever.chunks[i] for i in ids[: config.TOP_K_RERANK_INPUT]]
    if use_rerank and reranker is not None:
        return reranker.rerank(query, candidates, final_k)
    return candidates[:final_k]
