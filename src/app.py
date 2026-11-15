"""Step 7: interactive command-line app.

Run:  python -m src.app
      python -m src.app --mode dense --no-rerank     (to compare configurations)
"""
import argparse

from src.generate import generate_answer
from src.pipeline import search
from src.rerank import Reranker
from src.retrieve import HybridRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    print("Loading models...")
    retriever = HybridRetriever()
    reranker = None if args.no_rerank else Reranker()
    print(f"Ready. mode={args.mode}, rerank={not args.no_rerank}. Type 'exit' to quit.\n")

    while True:
        query = input("Question> ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break
        chunks = search(retriever, reranker, query, args.mode, not args.no_rerank)

        print("\n--- Retrieved chunks (debug view) ---")
        for i, c in enumerate(chunks, start=1):
            score = f" score={c['score']:.3f}" if "score" in c else ""
            print(f"[{i}] {c['source']} {c['id']}{score}\n    {c['text'][:150].replace(chr(10), ' ')}...")

        print("\n--- Answer ---")
        print(generate_answer(query, chunks), "\n")


if __name__ == "__main__":
    main()
