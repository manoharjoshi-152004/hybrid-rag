"""Step 8: compare dense / bm25 / hybrid / hybrid+rerank on your question set.

A retrieved chunk counts as correct if it comes from the expected source AND
contains the expected phrase. Edit eval/questions.json to match YOUR documents.
Results are printed and saved to eval/results.md.

Run:  python -m eval.run_eval
"""
import json
import re
from pathlib import Path

from src.pipeline import search
from src.rerank import Reranker
from src.retrieve import HybridRetriever

K = 5
HERE = Path(__file__).parent
CONFIGS = [
    ("Dense only", "dense", False),
    ("BM25 only", "bm25", False),
    ("Hybrid (RRF)", "hybrid", False),
    ("Hybrid + rerank", "hybrid", True),
]


def _norm(text: str) -> str:
    """Ignore case and any whitespace/newline differences (PDF text has odd line breaks)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: dict, q: dict) -> bool:
    return chunk["source"] == q["source"] and _norm(q["contains"]) in _norm(chunk["text"])


def main():
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    retriever, reranker = HybridRetriever(), Reranker()

    rows, all_misses = [], {}
    for name, mode, use_rerank in CONFIGS:
        hits, rr_sum, misses = 0, 0.0, []
        for q in questions:
            results = search(retriever, reranker, q["question"], mode, use_rerank, final_k=K)
            rank = next((i for i, c in enumerate(results, 1) if is_relevant(c, q)), None)
            if rank:
                hits += 1
                rr_sum += 1.0 / rank
            else:
                misses.append(q["question"])
        rows.append((name, hits / len(questions), rr_sum / len(questions)))
        all_misses[name] = misses

    lines = [
        f"# Evaluation results ({len(questions)} questions, top-{K})",
        "",
        f"| Configuration | Recall@{K} | MRR |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {r:.2f} | {m:.2f} |" for n, r, m in rows]
    for name, misses in all_misses.items():
        if misses:
            lines += ["", f"**Missed by {name}:**"] + [f"- {m}" for m in misses]
    report = "\n".join(lines) + "\n"

    print("\n" + report)
    (HERE / "results.md").write_text(report, encoding="utf-8")
    print(f"Saved to {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
