"""Chat page – conversational RAG interface with source citations."""

import streamlit as st

from services.chat_service import ChatService


def render():
    """Render the Chat page."""
    st.header("💬 Chat with your documents")

    chat_service = ChatService()
    datasets = chat_service.list_datasets()
    search_modes = chat_service.get_search_modes()
    search_mode_labels = {mode.key: mode.label for mode in search_modes}
    selected = None

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

        search_mode = st.radio(
            "Search Mode",
            options=[mode.key for mode in search_modes],
            format_func=search_mode_labels.get,
            key="chat_search_mode",
            help="\n".join(
                f"{mode.label}: {mode.help_text}" for mode in search_modes
            ),
        )

        st.divider()
        default_k = chat_service.get_search_mode(search_mode).default_top_k
        top_k = st.slider(
            "Sources to retrieve",
            2,
            10,
            default_k,
            key="chat_top_k",
            help="Raw PDF mode defaults to fewer sources to limit prompt size.",
        )

    state = _get_chat_state(selected)

    # ---- Display chat history ----
    for i, msg in enumerate(state["history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if i in state["sources"]:
                    _render_sources(state["sources"][i])
                if i in state["prompts"]:
                    _render_prompt(state["prompts"][i])

    # ---- Chat input ----
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        state["history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing..."):
                result = chat_service.ask(
                    user_input,
                    selected,
                    top_k=top_k,
                    mode=search_mode,
                )

            st.markdown(result["answer"])

            if result["sources"]:
                _render_sources(result["sources"])
            if result["prompt"]:
                _render_prompt(result["prompt"])

        # Save to history
        msg_index = len(state["history"])
        state["history"].append(
            {"role": "assistant", "content": result["answer"]}
        )
        if result["sources"]:
            state["sources"][msg_index] = result["sources"]
        if result["prompt"]:
            state["prompts"][msg_index] = result["prompt"]

    # ---- Sidebar: clear history ----
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            state["history"].clear()
            state["sources"].clear()
            state["prompts"].clear()
            st.rerun()


def _get_chat_state(selected_dataset: str):
    """Initialize and return chat session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_dataset" not in st.session_state:
        st.session_state.chat_dataset = selected_dataset
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}
    if "chat_prompts" not in st.session_state:
        st.session_state.chat_prompts = {}

    if st.session_state.chat_dataset != selected_dataset:
        st.session_state.chat_history = []
        st.session_state.chat_sources = {}
        st.session_state.chat_prompts = {}
        st.session_state.chat_dataset = selected_dataset

    return {
        "history": st.session_state.chat_history,
        "sources": st.session_state.chat_sources,
        "prompts": st.session_state.chat_prompts,
    }


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
