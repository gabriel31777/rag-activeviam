"""Chat page – conversational RAG interface with source citations."""

import streamlit as st

from core.rag_pipeline import RAGPipeline
from dataset.manager import DatasetManager


def render():
    """Render the Chat page."""
    st.header("💬 Chat with your documents")

    manager = DatasetManager()
    datasets = manager.list_datasets()

    # ---- Sidebar: dataset selection + settings ----
    with st.sidebar:
        st.subheader("Settings")
        if not datasets:
            st.warning(
                "No datasets available. Go to **Datasets** to create one."
            )
            return

        selected = st.selectbox(
            "Dataset",
            datasets,
            index=0,
            key="chat_dataset_select",
        )

        st.divider()
        
        # Search mode selection
        search_mode = st.radio(
            "Search Mode",
            options=["vector", "page_index", "pdf_raw"],
            format_func=lambda x: {
                "vector": "🔍 Vector Search (Hybrid + Reranking)",
                "page_index": "📑 Page Index (TOC Navigation)",
                "pdf_raw": "📄 PDF Raw (Direct Page Retrieval)"
            }[x],
            key="chat_search_mode",
            help="Vector: Advanced semantic search with BM25 and reranking\n"
                 "Page Index: Navigate document structure via table of contents\n"
                 "PDF Raw: Direct page-level retrieval without markdown conversion"
        )
        
        st.divider()
        # Adjust default based on mode to avoid token limits
        default_k = 3 if search_mode == "pdf_raw" else 5
        top_k = st.slider(
            "Sources to retrieve", 2, 10, default_k, key="chat_top_k",
            help="PDF Raw mode uses fewer sources by default to avoid token limits"
        )

    # ---- Session state init ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_dataset" not in st.session_state:
        st.session_state.chat_dataset = selected
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}
    if "chat_prompts" not in st.session_state:
        st.session_state.chat_prompts = {}

    # Reset history if dataset changed
    if st.session_state.chat_dataset != selected:
        st.session_state.chat_history = []
        st.session_state.chat_sources = {}
        st.session_state.chat_prompts = {}
        st.session_state.chat_dataset = selected

    # ---- Display chat history ----
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if i in st.session_state.chat_sources:
                    _render_sources(st.session_state.chat_sources[i])
                if i in st.session_state.chat_prompts:
                    _render_prompt(st.session_state.chat_prompts[i])

    # ---- Chat input ----
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing..."):
                pipeline = RAGPipeline()
                answer, sources, full_prompt = pipeline.query(
                    user_input, selected, top_k=top_k, mode=search_mode
                )

            st.markdown(answer)

            if sources:
                _render_sources(sources)
            if full_prompt:
                _render_prompt(full_prompt)

        # Save to history
        msg_index = len(st.session_state.chat_history)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        if sources:
            st.session_state.chat_sources[msg_index] = sources
        if full_prompt:
            st.session_state.chat_prompts[msg_index] = full_prompt

    # ---- Sidebar: clear history ----
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_sources = {}
            st.session_state.chat_prompts = {}
            st.rerun()


def _render_sources(sources):
    """Render source citations with expandable chunk text."""
    st.markdown("---")
    st.markdown("**📚 Sources used**")

    for i, src in enumerate(sources, 1):
        page_info = f" – Page {src['page']}" if src.get("page") else ""
        score_info = ""
        if src.get("rerank_score") is not None:
            score_info = f" (relevance: {src['rerank_score']:.2f})"

        label = f"[{i}] {src['source']}{page_info}{score_info}"

        with st.expander(label, expanded=False):
            st.markdown(src["text"])


def _render_prompt(prompt: str):
    """Render the full LLM prompt in a grey dropdown."""
    with st.expander("🔍 View prompt sent to LLM", expanded=False):
        st.markdown(
            f'<div style="color: #888; font-size: 0.85em; '
            f'white-space: pre-wrap; font-family: monospace; '
            f'background-color: #f5f5f5; padding: 12px; '
            f'border-radius: 6px; max-height: 400px; overflow-y: auto;">'
            f"{_escape_html(prompt)}</div>",
            unsafe_allow_html=True,
        )


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe rendering."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )
