"""Step 1: load raw files from data/ and clean the text."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir) -> list[dict]:
    """Return a list of {'text', 'source', 'page'} dicts (one per file, or per PDF page)."""
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                docs.append({"text": text, "source": path.name, "page": None})
        elif ext == ".pdf":
            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append({"text": text, "source": path.name, "page": page_no})
    return docs


if __name__ == "__main__":
    from src import config

    documents = load_documents(config.DATA_DIR)
    print(f"Loaded {len(documents)} document parts")
    for d in documents[:3]:
        print("-", d["source"], d["page"], d["text"][:80].replace("\n", " "), "...")
