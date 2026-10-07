"""
Knowledge Base File Explorer and Document Inspector Component.

Displays the active documents loaded in PostgreSQL, allows inspecting original
document content (PDF, Markdown, TXT), inspecting indexed chunks, and managing documents.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import html
import streamlit as st

from app.config import RAW_DATA_DIR
from app.db import (
    get_loaded_documents,
    delete_document,
    get_document_chunks,
)
from ui.ingestion_helper import (
    ingest_document_file,
    preload_sample_documents,
)


def _get_file_icon(doc_type: str) -> str:
    """Returns an icon based on document file type."""
    dtype = (doc_type or "").lower().strip()
    if dtype in ("pdf", ".pdf"):
        return "📘"
    elif dtype in ("md", ".md", "markdown"):
        return "📝"
    else:
        return "📄"


def _read_original_document(filename: str, document_id: int) -> Dict[str, Any]:
    """
    Reads original file content from data/raw/ or reconstructs it from stored chunks.
    """
    raw_path = RAW_DATA_DIR / filename
    if raw_path.exists():
        ext = raw_path.suffix.lower()
        if ext in (".md", ".txt"):
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    return {"type": ext.lstrip("."), "content": f.read(), "source": "disk", "path": str(raw_path)}
            except UnicodeDecodeError:
                with open(raw_path, "r", encoding="latin-1") as f:
                    return {"type": ext.lstrip("."), "content": f.read(), "source": "disk", "path": str(raw_path)}
        elif ext == ".pdf":
            # Extract text from PDF for inline reading
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(raw_path))
                pages = []
                for pno in range(len(doc)):
                    pages.append({"page": pno + 1, "text": doc[pno].get_text()})
                doc.close()
                return {"type": "pdf", "pages": pages, "source": "disk", "path": str(raw_path)}
            except Exception:
                pass

    # Fallback: Reconstruct document from indexed chunks in database
    chunks = get_document_chunks(document_id)
    combined = "\n\n".join(c.get("content", "") for c in chunks)
    return {"type": Path(filename).suffix.lstrip(".") or "text", "content": combined, "source": "database_chunks"}


def render_document_management() -> None:
    """
    Renders top-level document management, file explorer, and file uploader.
    """
    with st.expander("📁 Knowledge Base & Document Management", expanded=False):
        tab_explorer, tab_upload, tab_sample = st.tabs([
            "📂 Active Database Files",
            "📤 Ingest New Document",
            "⚡ Preload Sample Corpus",
        ])

        # ----------------------------------------------------------------------
        # Tab 1: Active Database Files Explorer
        # ----------------------------------------------------------------------
        with tab_explorer:
            try:
                docs = get_loaded_documents()
            except Exception as e:
                st.warning(f"Unable to query PostgreSQL documents: {e}")
                docs = []

            total_docs = len(docs)
            total_chunks = sum(d.get("chunk_count", 0) for d in docs)

            # Metrics row
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Loaded Documents", f"{total_docs} files")
            with col_m2:
                st.metric("Indexed Chunks", f"{total_chunks} chunks")
            with col_m3:
                st.metric("Storage Engine", "PostgreSQL + pgvector")

            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

            if not docs:
                st.info(
                    "**The database is currently empty.** "
                    "Upload a document in the 'Ingest New Document' tab or click "
                    "'Preload Sample Corpus' to test the system out of the box."
                )
            else:
                # File selection dropdown for detailed inspection
                doc_options = {
                    f"{_get_file_icon(d['document_type'])} {d['filename']} (ID: {d['id']}, {d['chunk_count']} chunks)": d
                    for d in docs
                }
                selected_label = st.selectbox(
                    "Select a document to inspect original content:",
                    options=list(doc_options.keys()),
                    help="Choose an active document to view its original full text and indexed chunks.",
                )

                if selected_label:
                    sel_doc = doc_options[selected_label]
                    doc_id = sel_doc["id"]
                    filename = sel_doc["filename"]
                    doc_type = sel_doc.get("document_type", "txt")
                    dept = sel_doc.get("department", "public")
                    clearance = sel_doc.get("access_level", "public")
                    chunks_count = sel_doc.get("chunk_count", 0)
                    created_at = sel_doc.get("created_at")

                    # Metadata badge container
                    st.markdown(
                        f"""
                        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: 1.25rem; font-size: 0.88rem; color: #0f172a;">
                            <div><b>Document ID:</b> <code>{doc_id}</code></div>
                            <div><b>Format:</b> <code>.{html.escape(str(doc_type).upper())}</code></div>
                            <div><b>Department:</b> <span style="background: #e2e8f0; padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 600;">{html.escape(dept)}</span></div>
                            <div><b>Clearance:</b> <span style="background: #e2e8f0; padding: 0.15rem 0.45rem; border-radius: 4px; font-weight: 600;">{html.escape(clearance)}</span></div>
                            <div><b>Chunks:</b> <b>{chunks_count}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Action buttons
                    col_del, col_info = st.columns([1.5, 4])
                    with col_del:
                        if st.button(f"🗑️ Delete Document", key=f"del_doc_{doc_id}", type="secondary"):
                            try:
                                delete_document(doc_id)
                                st.success(f"Document '{filename}' deleted from database.")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Failed to delete: {ex}")

                    # View original content
                    with st.expander(f"📖 View Original Document: {filename}", expanded=True):
                        doc_content = _read_original_document(filename, doc_id)

                        if doc_content.get("type") == "pdf" and "pages" in doc_content:
                            pages = doc_content["pages"]
                            st.caption(f"Showing extracted content from {len(pages)} PDF pages:")
                            for p in pages:
                                st.markdown(f"**Page {p['page']}**")
                                st.text_area(
                                    label=f"Page {p['page']} Text",
                                    value=p["text"],
                                    height=180,
                                    key=f"pdf_page_{doc_id}_{p['page']}",
                                    disabled=True,
                                )
                            # Offer PDF download button if file is on disk
                            pdf_path = RAW_DATA_DIR / filename
                            if pdf_path.exists():
                                with open(pdf_path, "rb") as pf:
                                    st.download_button(
                                        label=f"⬇️ Download Original {filename}",
                                        data=pf.read(),
                                        file_name=filename,
                                        mime="application/pdf",
                                    )
                        elif doc_content.get("type") in ("md", "markdown"):
                            st.markdown(doc_content.get("content", ""))
                        else:
                            st.text_area(
                                label="Document Text",
                                value=doc_content.get("content", ""),
                                height=260,
                                disabled=True,
                            )

                    # Inspect indexed chunks
                    with st.expander(f"🧩 Inspect Indexed Chunks ({chunks_count} passages)", expanded=False):
                        chunks = get_document_chunks(doc_id)
                        for c in chunks:
                            c_idx = c.get("chunk_index", 0)
                            p_num = c.get("page_number")
                            p_str = f"Page {p_num}" if p_num else "Page N/A"
                            sec = c.get("section")
                            sec_str = f" • Section: {sec}" if sec else ""
                            st.markdown(f"**Chunk #{c_idx} ({p_str}{sec_str})**")
                            st.markdown(
                                f"<div style='background: #f8fafc; border-left: 3px solid #3b82f6; padding: 0.5rem 0.75rem; border-radius: 4px; color: #0f172a; font-size: 0.88rem; margin-bottom: 0.75rem;'>{html.escape(c.get('content', ''))}</div>",
                                unsafe_allow_html=True,
                            )

        # ----------------------------------------------------------------------
        # Tab 2: Upload & Ingest New Document
        # ----------------------------------------------------------------------
        with tab_upload:
            st.markdown(
                "<div style='font-size: 0.9rem; color: #334155; margin-bottom: 0.5rem; font-weight: 600;'>"
                "Upload and index any PDF, Markdown, or Plain Text document into PostgreSQL:</div>",
                unsafe_allow_html=True,
            )
            up_file = st.file_uploader(
                "Select File to Ingest",
                type=["pdf", "md", "txt"],
                help="Supported extensions: .pdf, .md, .txt",
                key="tab_doc_uploader",
            )
            col_d, col_c = st.columns(2)
            with col_d:
                doc_dept = st.selectbox(
                    "Assign Department",
                    options=["public", "engineering", "finance", "hr"],
                    index=0,
                    key="tab_up_dept",
                )
            with col_c:
                doc_clearance = st.selectbox(
                    "Assign Clearance Level",
                    options=["public", "employee", "manager", "admin"],
                    index=0,
                    key="tab_up_clearance",
                )
            extract_kg_opt = st.checkbox(
                "Extract Knowledge Graph Entities & Relationships (LLM)",
                value=False,
                key="tab_up_kg",
                help="Uses Gemini to extract entities and triples into the PostgreSQL graph tables.",
            )

            if up_file is not None:
                if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
                    with st.spinner(f"Ingesting '{up_file.name}', generating 384-d MiniLM vectors, and indexing..."):
                        try:
                            res = ingest_document_file(
                                file_bytes=up_file.getvalue(),
                                filename=up_file.name,
                                department=doc_dept,
                                access_level=doc_clearance,
                                extract_graph=extract_kg_opt,
                            )
                            st.success(
                                f"**Successfully Indexed '{res['filename']}'!**\n\n"
                                f"- Document ID: `{res['document_id']}`\n"
                                f"- Stored Chunks: `{res['chunks_stored']}`\n"
                                f"- Vector Dimension: `{res['vector_dimension']}` (MiniLM)\n"
                                + (f"- Graph Relationships: `{res['graph_stats'].get('relationships_inserted', 0)}` triples\n" if extract_kg_opt else "")
                            )
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Ingestion failed: {ex}")

        # ----------------------------------------------------------------------
        # Tab 3: Preload Sample Corpus
        # ----------------------------------------------------------------------
        with tab_sample:
            st.markdown(
                "<div style='font-size: 0.9rem; color: #334155; margin-bottom: 0.5rem;'>"
                "Load the 3 bundled license-free enterprise documents into PostgreSQL so an interviewer or tester can immediately explore dual-engine retrieval:</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                1. **`support_faq.txt`** — Department: `public` • Clearance: `public`
                2. **`enterprise_platform_architecture.pdf`** — Department: `engineering` • Clearance: `employee`
                3. **`ai_governance_policy.md`** — Department: `engineering` • Clearance: `manager`
                """
            )
            extract_sample_kg = st.checkbox(
                "Also extract Knowledge Graph triples during preload (requires Gemini API key)",
                value=False,
                key="sample_kg_opt",
            )
            if st.button("⚡ Preload Sample Enterprise Documents", type="primary"):
                with st.spinner("Preloading and embedding sample enterprise documents into PostgreSQL..."):
                    try:
                        results = preload_sample_documents(extract_graph=extract_sample_kg)
                        st.success(f"Successfully loaded {len(results)} sample enterprise documents into the database!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Preload failed: {ex}")
