"""Datasets management page – create, delete, upload, and index datasets."""

import streamlit as st

from core.index_metadata import IndexMetadata
from core.rag_pipeline import RAGPipeline
from core.vector_store import VectorStore
from dataset.loaders import SUPPORTED_EXTENSIONS
from dataset.manager import DatasetManager
from utils.config import get_config


def render():
    """Render the Datasets management page."""
    st.header("📂 Datasets")

    manager = DatasetManager()

    # ---- Create dataset ----
    st.subheader("Create a new dataset")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_name = st.text_input(
            "Dataset name",
            placeholder="e.g. research_papers",
            label_visibility="collapsed",
        )
    with col2:
        create_btn = st.button("➕ Create", use_container_width=True)

    if create_btn and new_name:
        try:
            manager.create_dataset(new_name)
            st.success(f"Dataset **{new_name}** created.")
            st.rerun()
        except FileExistsError:
            st.warning(f"Dataset **{new_name}** already exists.")
        except ValueError as e:
            st.error(str(e))

    st.divider()

    # ---- List datasets ----
    datasets = manager.list_datasets()
    if not datasets:
        st.info("No datasets yet. Create one above to get started.")
        return

    for ds in datasets:
        with st.expander(f"📁 **{ds}**", expanded=False):
            _render_dataset_panel(ds, manager)


def _render_dataset_panel(
    dataset: str,
    manager: DatasetManager,
):
    """Render the management panel for a single dataset."""
    documents = manager.list_documents(dataset)

    # -- Upload documents --
    st.markdown("**Upload documents**")
    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        key=f"upload_{dataset}",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            try:
                manager.save_uploaded_file(dataset, uf.name, uf.read())
                st.success(f"Uploaded **{uf.name}**")
            except ValueError as e:
                st.error(str(e))
        st.rerun()

    # -- List documents --
    if documents:
        st.markdown("**Documents**")
        for doc in documents:
            col_doc, col_del = st.columns([5, 1])
            with col_doc:
                st.text(f"  📄 {doc}")
            with col_del:
                if st.button("🗑️", key=f"del_{dataset}_{doc}"):
                    manager.delete_document(dataset, doc)
                    st.rerun()
    else:
        st.caption("No documents uploaded yet.")

    st.markdown("---")

    # -- Index info --
    config = get_config()
    try:
        vs = VectorStore(dataset)
        chunk_count = vs.count()
        st.caption(f"Vector index: **{chunk_count}** chunks indexed")
        
        # Check index compatibility
        metadata = IndexMetadata(dataset, config.vectordb_path)
        status_info = metadata.get_status_info()
        
        # Display compatibility status
        if status_info["needs_rebuild"]:
            st.warning(
                f"{status_info['status']}: {status_info['message']}",
                icon="⚠️"
            )
        else:
            st.success(status_info["message"], icon="✅")
            
    except Exception:
        chunk_count = 0
        st.caption("Vector index: not built yet")

    # -- Actions --
    col_build, col_delete = st.columns(2)

    with col_build:
        if st.button(
            "🔄 Build / Rebuild Index",
            key=f"build_{dataset}",
            use_container_width=True,
        ):
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def _progress(step, total, message):
                if total > 0:
                    progress_bar.progress(
                        step / total,
                        text=f"({step}/{total}) {message}",
                    )
                status_text.caption(message)

            try:
                pipeline = RAGPipeline()
            except ValueError as e:
                st.error(str(e))
                return
            n = pipeline.ingest_dataset(dataset, progress_callback=_progress)
            progress_bar.progress(1.0, text="Complete!")
            st.success(f"Index built: **{n}** chunks")
            st.rerun()

    with col_delete:
        if st.button(
            "🗑️ Delete Dataset",
            key=f"delete_{dataset}",
            type="primary",
            use_container_width=True,
        ):
            manager.delete_dataset(dataset)
            st.success(f"Dataset **{dataset}** deleted.")
            st.rerun()
