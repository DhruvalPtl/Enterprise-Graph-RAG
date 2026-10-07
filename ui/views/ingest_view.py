"""
Ingest View Component.

Full-page document ingestion interface to upload, chunk, embed, and index
PDF, Markdown, and Plain Text files into PostgreSQL pgvector and Knowledge Graph.
"""
from typing import Dict, Any
import streamlit as st

from ui.ingestion_helper import ingest_document_file


def render_ingest_view() -> None:
    """
    Renders the full-page Document Ingestion interface.
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
                Upload & Ingest Document
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-top: 0.2rem;">
                Parse, chunk, embed, and index new documents into PostgreSQL pgvector and Knowledge Graph.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 0.25rem;">
                    Supported Formats: PDF (.pdf), Markdown (.md), Plain Text (.txt)
                </div>
                <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                    The pipeline performs structure-aware chunking (preserving headings and page numbers), computes 384-dimensional dense vectors using MiniLM, and stores them in PostgreSQL with an HNSW cosine index.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select Document File",
            type=["pdf", "md", "txt"],
            help="Choose a PDF, Markdown, or text file from your computer.",
            key="page_ingest_uploader",
        )

        col_dept, col_clearance = st.columns(2)
        with col_dept:
            department = st.selectbox(
                "Assign Organization Department",
                options=["public", "engineering", "finance", "hr"],
                index=0,
                help="Restricts document retrieval to users belonging to this department.",
                key="page_ingest_dept",
            )
        with col_clearance:
            clearance = st.selectbox(
                "Assign Access Clearance Level",
                options=["public", "employee", "manager", "admin"],
                index=0,
                help="Minimum security clearance level required to retrieve this document.",
                key="page_ingest_clearance",
            )

        extract_graph = st.checkbox(
            "Extract Knowledge Graph Entities & Relationships (LLM)",
            value=False,
            help="Calls Gemini to extract canonical entities and directed relationship triples into PostgreSQL graph tables.",
            key="page_ingest_kg",
        )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if extract_graph and not api_key:
            st.warning("⚠️ **Gemini API Key Required:** Enter your Gemini API Key in the sidebar Control Panel to enable Knowledge Graph extraction.")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
                with st.spinner(f"Ingesting '{uploaded_file.name}', computing MiniLM embeddings, and persisting to PostgreSQL..."):
                    try:
                        res = ingest_document_file(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            department=department,
                            access_level=clearance,
                            extract_graph=extract_graph,
                        )
                        graph_stats = res.get("graph_stats", {})
                        st.success(
                            f"**Ingestion Complete! Document is now live and queryable.**\n\n"
                            f"* **Filename:** `{res['filename']}`\n"
                            f"* **Document ID:** `{res['document_id']}`\n"
                            f"* **Passage Chunks Stored:** `{res['chunks_stored']}`\n"
                            f"* **Vector Dimension:** `{res['vector_dimension']}` (MiniLM pgvector)\n"
                            f"* **Department / Clearance:** `{res['department']}` / `{res['access_level']}`\n"
                            + (f"* **Graph Triples Extracted:** `{graph_stats.get('relationships_inserted', 0)}` relationships (`{graph_stats.get('entities_upserted', 0)}` entities)\n" if extract_graph else "")
                        )
                        if graph_stats.get("error"):
                            st.warning(f"⚠️ **Knowledge Graph Notice:** {graph_stats['error']}")
                    except Exception as ex:
                        st.error(f"Ingestion failed: {ex}")
