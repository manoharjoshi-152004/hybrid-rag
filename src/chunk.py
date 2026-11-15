"""Step 2: split documents into overlapping chunks, keeping metadata."""
import re


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries, hard-cutting if needed."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:                 # a single giant sentence
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= size else _split_long_paragraph(p, size))

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:                          # start overlap at a word boundary
                tail = tail.split(" ", 1)[1]
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    for doc in docs:
        for text in split_text(doc["text"], size, overlap):
            chunks.append({
                "id": f"c{len(chunks):05d}",
                "text": text,
                "source": doc["source"],
                "page": doc["page"],
            })
    return chunks


if __name__ == "__main__":
    from src import config
    from src.ingest import load_documents

    docs = load_documents(config.DATA_DIR)
    chunks = chunk_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"{len(chunks)} chunks. Inspect a few by eye:\n")
    for c in chunks[:5]:
        print(f"[{c['id']}] {c['source']} p.{c['page']} ({len(c['text'])} chars)\n{c['text']}\n{'-' * 40}")
