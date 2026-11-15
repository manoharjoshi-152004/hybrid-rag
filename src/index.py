"""Step 3: build and save the dense index. (BM25 is rebuilt at load time; it is fast.)

Run:  python -m src.index
"""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config
from src.chunk import chunk_documents
from src.ingest import load_documents


def build_index():
    docs = load_documents(config.DATA_DIR)
    if not docs:
        raise SystemExit(f"No documents found in {config.DATA_DIR}. Add .txt/.md/.pdf files.")
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"{len(docs)} document parts -> {len(chunks)} chunks")

    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,   # so dot product == cosine similarity
        show_progress_bar=True,
    )

    config.INDEX_DIR.mkdir(exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", embeddings)
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    build_index()
