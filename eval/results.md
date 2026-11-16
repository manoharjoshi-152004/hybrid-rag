# Evaluation results (33 questions, top-5)

| Configuration | Recall@5 | MRR |
|---|---|---|
| Dense only | 0.88 | 0.74 |
| BM25 only | 0.82 | 0.71 |
| Hybrid (RRF) | 0.91 | 0.77 |
| Hybrid + rerank | 0.94 | 0.86 |

**Missed by Dense only:**
- Which approach reviews its own answer and searches again if it is weak?
- What example question illustrates Corrective RAG?
- What topic does the Attention-based RAG example ask about?
- When is Adaptive RAG the best choice?

**Missed by BM25 only:**
- Which approach answers easy questions without searching at all?
- What happens when a search returns documents that do not fit the question?
- How does a RAG system keep spending low?
- Which two companies are compared in the Agentic RAG example?
- What example question illustrates Corrective RAG?
- When is Adaptive RAG the best choice?

**Missed by Hybrid (RRF):**
- What example question illustrates Corrective RAG?
- What topic does the Attention-based RAG example ask about?
- When is Adaptive RAG the best choice?

**Missed by Hybrid + rerank:**
- Which approach reviews its own answer and searches again if it is weak?
- What topic does the Attention-based RAG example ask about?
