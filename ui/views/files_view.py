"""
Knowledge Base Files View.

Full-page file explorer to view, inspect, read, and manage documents currently loaded
in the PostgreSQL database.
"""
from pathlib import Path
from typing import Dict, Any, List
import html
import streamlit as st

from app.config import RAW_DATA_DIR
from app.db import (
    get_loaded_documents,
    delete_document,
    get_document_chunks,
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
    """Reads original file content from data/raw/ or reconstructs it from stored chunks."""
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
            try:
                import fitz
                doc = fitz.open(str(raw_path))
                pages = []
                for pno in range(len(doc)):
                    pages.append({"page": pno + 1, "text": doc[pno].get_text()})
                doc.close()
                return {"type": "pdf", "pages": pages, "source": "disk", "path": str(raw_path)}
            except Exception:
                pass

    # Fallback to database chunks
    chunks = get_document_chunks(document_id)
    combined = "\n\n".join(c.get("content", "") for c in chunks)
    return {"type": Path(filename).suffix.lstrip(".") or "text", "content": combined, "source": "database_chunks"}


def render_files_view() -> None:
    """
    Renders the full-page Knowledge Base File Explorer.
    """
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
                Knowledge Base Files
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-top: 0.2rem;">
                Explore, inspect original text, and manage documents currently loaded in PostgreSQL.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        docs = get_loaded_documents()
    except Exception as e:
        st.warning(f"Unable to query PostgreSQL documents: {e}")
        docs = []

    total_docs = len(docs)
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)

    # 1. Summary Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Documents", f"{total_docs} files")
    with m2:
        st.metric("Indexed Chunks", f"{total_chunks} passages")
    with m3:
        st.metric("Storage Engine", "PostgreSQL + pgvector")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    if not docs:
        st.info(
            "**No documents are currently loaded in the database.** "
            "Navigate to the **'Upload & Ingest'** page in the sidebar to add your first document."
        )
        return

    # 2. Select document to inspect
    st.markdown("### Select Document to Inspect")
    doc_options = {
        f"{_get_file_icon(d['document_type'])} {d['filename']} (ID: {d['id']}, {d['chunk_count']} chunks, Dept: {d['department']})": d
        for d in docs
    }
    selected_label = st.selectbox(
        "Choose an active document:",
        options=list(doc_options.keys()),
        help="Select any document to view its full content, metadata, and indexed chunk passages.",
    )

    if selected_label:
        sel_doc = doc_options[selected_label]
        doc_id = sel_doc["id"]
        filename = sel_doc["filename"]
        doc_type = sel_doc.get("document_type", "txt")
        dept = sel_doc.get("department", "public")
        clearance = sel_doc.get("access_level", "public")
        chunks_count = sel_doc.get("chunk_count", 0)

        # Metadata Card
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 0.5rem; margin-bottom: 1.25rem; display: flex; flex-wrap: wrap; gap: 1.5rem; font-size: 0.9rem; color: #0f172a;">
                <div><b>Document ID:</b> <code>{doc_id}</code></div>
                <div><b>Format:</b> <code>.{html.escape(str(doc_type).upper())}</code></div>
                <div><b>Department:</b> <span style="background: #e2e8f0; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">{html.escape(dept)}</span></div>
                <div><b>Clearance:</b> <span style="background: #e2e8f0; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 700;">{html.escape(clearance)}</span></div>
                <div><b>Indexed Chunks:</b> <b>{chunks_count}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_actions, _ = st.columns([2, 4])
        with col_actions:
            if st.button(f"🗑️ Delete '{filename}' from Database", type="secondary", use_container_width=True):
                try:
                    delete_document(doc_id)
                    st.success(f"Document '{filename}' deleted from database.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to delete: {ex}")

        # 3. View original document content
        st.markdown(f"#### 📖 Original Content: `{filename}`")
        doc_content = _read_original_document(filename, doc_id)

        if doc_content.get("type") == "pdf" and "pages" in doc_content:
            pages = doc_content["pages"]
            st.caption(f"Extracted text from {len(pages)} PDF pages:")
            for p in pages:
                st.markdown(f"**Page {p['page']}**")
                st.text_area(
                    label=f"Page {p['page']} Text",
                    value=p["text"],
                    height=180,
                    key=f"file_view_pdf_{doc_id}_{p['page']}",
                    disabled=True,
                )
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
                label="Document Text Content",
                value=doc_content.get("content", ""),
                height=300,
                disabled=True,
            )

        # 4. View indexed chunks
        st.markdown(f"#### 🧩 Indexed Chunk Passages ({chunks_count} chunks)")
        chunks = get_document_chunks(doc_id)
        for c in chunks:
            c_idx = c.get("chunk_index", 0)
            p_num = c.get("page_number")
            p_str = f"Page {p_num}" if p_num else "Page N/A"
            sec = c.get("section")
            sec_str = f" • Section: {sec}" if sec else ""
            with st.expander(f"Chunk #{c_idx} ({p_str}{sec_str})", expanded=False):
                st.markdown(
                    f"<div style='background: #f8fafc; border-left: 3px solid #3b82f6; padding: 0.6rem 0.8rem; border-radius: 4px; color: #0f172a; font-size: 0.9rem;'>{html.escape(c.get('content', ''))}</div>",
                    unsafe_allow_html=True,
                )
