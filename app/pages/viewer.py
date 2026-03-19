"""Document Viewer page – inspect extracted content of uploaded documents."""

import os

import streamlit as st

from dataset.loaders import load_document
from dataset.manager import DatasetManager


def render():
    """Render the Document Viewer page."""
    st.header("📖 Document Viewer")

    manager = DatasetManager()
    datasets = manager.list_datasets()

    if not datasets:
        st.info("No datasets yet. Go to **Datasets** to create one.")
        return

    # ---- Sidebar: dataset + document selection ----
    with st.sidebar:
        st.subheader("Document")
        selected_ds = st.selectbox(
            "Dataset",
            datasets,
            key="viewer_dataset",
        )

    documents = manager.list_documents(selected_ds)
    if not documents:
        st.info(f"No documents in **{selected_ds}**. Upload some first.")
        return

    with st.sidebar:
        selected_doc = st.selectbox(
            "Document",
            documents,
            key="viewer_document",
        )

    file_path = os.path.join(
        manager.get_dataset_path(selected_ds), selected_doc
    )

    # ---- View mode ----
    view_mode = st.radio(
        "View mode",
        ["Rendered", "Raw text"],
        horizontal=True,
        key="viewer_mode",
    )

    st.divider()

    # ---- Load and display ----
    try:
        doc = load_document(file_path)
    except Exception as e:
        st.error(f"Failed to load document: {e}")
        return

    pages = doc.get("pages")

    if pages:
        # PDF: per-page view
        total_pages = len(pages)
        with st.sidebar:
            st.subheader("Navigation")
            show_all = st.checkbox(
                "Show all pages", value=False, key="viewer_show_all"
            )
            if not show_all:
                page_num = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key="viewer_page",
                )

        st.caption(f"**{selected_doc}** — {total_pages} pages")

        if show_all:
            for p in pages:
                st.subheader(f"Page {p['page']}")
                _display_text(p["text"], view_mode)
                st.divider()
        else:
            page_data = next(
                (p for p in pages if p["page"] == page_num), None
            )
            if page_data:
                st.subheader(f"Page {page_num} / {total_pages}")
                _display_text(page_data["text"], view_mode)
            else:
                st.warning(f"Page {page_num} has no extractable text.")
    else:
        # Non-PDF: single view
        st.caption(f"**{selected_doc}**")
        _display_text(doc["text"], view_mode)


def _display_text(text: str, mode: str):
    """Show text either rendered or as raw source."""
    if mode == "Rendered":
        st.markdown(text, unsafe_allow_html=False)
    else:
        st.code(text, language="text")
