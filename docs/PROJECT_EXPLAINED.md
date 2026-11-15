# Project explained (for interviews, reviews, and your own understanding)

## 1. The problem
LLMs don't know your private documents and can make things up. **Retrieval-Augmented Generation (RAG)** fixes this: first *find* the relevant passages, then ask the LLM to answer *using only those passages*.

## 2. The pipeline in plain English
1. **Ingest:** read files, clean the text, remember the filename/page.
2. **Chunk:** cut text into ~600-character pieces with 15% overlap so answers aren't split across a boundary.
3. **Index:** (a) turn each chunk into a vector of numbers that captures its meaning (embeddings); (b) build a keyword index (BM25).
4. **Retrieve:** for a question, get the top 30 chunks from each index.
5. **Fuse:** combine the two lists with Reciprocal Rank Fusion.
6. **Rerank:** a slower, more accurate model re-scores the top 20 and keeps the best 4.
7. **Generate:** Gemini answers from those 4 chunks and cites them as [1], [2].

## 3. Key concepts

**Embeddings (dense retrieval).** A model maps text to a vector; similar meanings land close together. Finds "how do I reset the device" even if the text says "restore factory settings". Weak on exact identifiers like `E-4021`.

**BM25 (sparse retrieval).** Classic keyword scoring: rewards rare words that appear often in a chunk. Excellent for codes, names, acronyms. Blind to synonyms.

**Reciprocal Rank Fusion (RRF).** Each chunk gets `1/(60 + rank)` from every list it appears in; scores are summed.
Example: chunk A is #1 in dense and #3 in BM25 → 1/61 + 1/63 = 0.03226. Chunk B is #2 in dense and #1 in BM25 → 1/62 + 1/61 = 0.03252. B wins because it is strong in both. Works on ranks, so no score normalisation is needed.

**Bi-encoder vs cross-encoder.** Embeddings are a bi-encoder: query and passage are encoded separately, so passage vectors are precomputed (fast, coarse). A cross-encoder reads query and passage *together* (slow, accurate). So: cheap methods gather candidates, the cross-encoder picks winners.

**Grounded generation.** The prompt tells the model to use only the supplied context and say "I don't know" otherwise, which reduces hallucination and makes answers checkable via citations.

## 4. How it was evaluated
- A fixed set of questions, each with the source file and a phrase that must appear in the correct chunk.
- **Recall@5:** fraction of questions where a correct chunk is in the top 5.
- **MRR:** average of 1/rank of the first correct chunk (rank 1 → 1.0, rank 2 → 0.5, not found → 0). For ranks 1, 2, and not found: (1 + 0.5 + 0) / 3 = 0.5.
- Four configurations compared: dense, BM25, hybrid, hybrid + rerank.

## 5. Trade-offs and honest limitations
- Reranking adds latency (runs a model per candidate). Mitigation: rerank only 20, use a smaller reranker.
- Chunk size is a trade-off: small chunks are precise but lose context, large chunks add noise.
- The sample dataset is tiny, so results are only meaningful on a larger corpus.
- Free-tier Gemini has rate limits and may use inputs to improve Google products, so don't use confidential documents.

## 6. Likely interview questions
1. **Why not just use embeddings?** They miss exact tokens such as IDs and error codes; BM25 covers that.
2. **Why RRF instead of weighted scores?** Cosine and BM25 scores aren't comparable; ranks are.
3. **Why rerank only the top 20?** Cross-encoders are too slow to run on the whole corpus.
4. **How do you know it works?** Fixed eval set, Recall@k and MRR, four-way comparison.
5. **What if the answer isn't in the documents?** The prompt forces "I don't know"; I tested with an out-of-scope question.
6. **How would you scale it?** Move to a vector DB (Qdrant/Weaviate/pgvector), persist BM25 in a search engine, cache embeddings, batch rerank on GPU.
7. **What causes wrong answers most often?** Retrieval or chunking, not the LLM. I inspect the retrieved chunks first.
8. **Why overlap chunks?** So a fact straddling a boundary still appears whole in one chunk.
9. **Why the query prefix for BGE?** The model was trained with an instruction on queries; documents are embedded without it.
10. **What would you improve next?** Query rewriting, metadata filters, contextual chunk headers, parent-document retrieval, table-aware PDF parsing.

## 7. 30-second pitch
"I built a RAG system that combines embedding search and BM25 keyword search, fuses them with Reciprocal Rank Fusion, reranks with a cross-encoder, and answers with Gemini using citations. I evaluated four configurations with Recall@5 and MRR to show what each stage adds."
