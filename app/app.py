"""RAG Application – main Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide deploy button
st.markdown(
    """
    <style>
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Navigation ----
PAGES = {
    "💬 Chat": "chat",
    "📂 Datasets": "datasets",
}

with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.caption("Hybrid Search + Cross-Encoder Reranking")
    st.divider()
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

# ---- Routing ----
selected = PAGES[page]

if selected == "chat":
    from pages.chat import render

    render()
elif selected == "datasets":
    from pages.datasets import render

    render()
