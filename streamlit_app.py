"""Web UI.  Run:  streamlit run streamlit_app.py"""
import streamlit as st

from src import config
from src.generate import generate_answer
from src.pipeline import search
from src.rerank import Reranker
from src.retrieve import HybridRetriever

st.set_page_config(page_title="Hybrid RAG + Reranking", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run takes a while)...")
def load_models():
    return HybridRetriever(), Reranker()


retriever, reranker = load_models()

st.title("🔎 Hybrid RAG + Reranking")
st.caption("Dense + BM25 retrieval → RRF fusion → cross-encoder rerank → Gemini answer with citations")

with st.sidebar:
    st.header("Pipeline settings")
    mode = st.radio("Retrieval mode", ["hybrid", "dense", "bm25"], index=0)
    use_rerank = st.checkbox("Use reranker", value=True)
    final_k = st.slider("Chunks sent to the LLM", 1, 8, config.TOP_K_FINAL)
    st.markdown(f"**LLM:** `{config.LLM_MODEL}`  \n**Chunks indexed:** {len(retriever.chunks)}")

query = st.text_input("Ask a question about your documents", placeholder="e.g. What does error E-4021 mean?")

if query:
    with st.spinner("Retrieving..."):
        chunks = search(retriever, reranker, query, mode, use_rerank, final_k)
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, chunks)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources used")
    for i, c in enumerate(chunks, start=1):
        score = f" · rerank score {c['score']:.3f}" if "score" in c else ""
        page = f" · p.{c['page']}" if c.get("page") else ""
        with st.expander(f"[{i}] {c['source']}{page}{score}"):
            st.write(c["text"])
