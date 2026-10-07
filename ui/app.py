"""
Enterprise Graph RAG - Interactive Web Application.

A professional Streamlit user interface built on top of the Enterprise RAG FastAPI backend.
Communicates retrieval source provenance (Dense Vector, BM25, Knowledge Graph),
transparent source citations, and pipeline execution telemetry.
"""
import os
import sys
from pathlib import Path

# Add project root to python path to guarantee clean module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from ui import api_client
from ui.components.header import render_header
from ui.components.query_input import render_query_input
from ui.components.answer_view import render_answer
from ui.components.citations_view import render_citations
from ui.components.graph_evidence_view import render_graph_evidence
from ui.components.diagnostics_view import render_diagnostics

# Page configuration
st.set_page_config(
    page_title="Enterprise Graph RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
css_file = Path(__file__).parent / "styles" / "main.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">Control Panel</div>',
    unsafe_allow_html=True,
)

default_api_url = os.getenv("RAG_API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input(
    "FastAPI Service URL",
    value=default_api_url,
    help="Target FastAPI endpoint hosting POST /query and GET /health.",
)

# Immediate health check
health_info = api_client.check_health(api_url)

st.sidebar.markdown("---")
st.sidebar.markdown("### Retrieval Engine")

retrieval_mode = st.sidebar.radio(
    "Engine Mode",
    options=["hybrid_graph_vector", "vector_bm25_rrf"],
    format_func=lambda m: (
        "⚡ Hybrid (Vector + BM25 + Graph)"
        if m == "hybrid_graph_vector"
        else "🔍 Baseline (Vector + BM25 only)"
    ),
    help="Select whether to traverse PostgreSQL knowledge graph entities and relationships in addition to dense & lexical retrieval.",
)

top_k = st.sidebar.slider(
    "Top K Citations",
    min_value=1,
    max_value=20,
    value=5,
    help="Final reranked evidence passages provided to the grounded context.",
)

candidate_k = st.sidebar.slider(
    "Candidate Pool K",
    min_value=5,
    max_value=50,
    value=20,
    help="Stage 1 retrieval pool limit before cross-encoder reranking.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Access Control & RBAC")

dept = st.sidebar.selectbox(
    "Department",
    options=["public", "engineering", "finance", "hr"],
    index=0,
    help="Caller organizational department for metadata filtering.",
)

access_lvl = st.sidebar.selectbox(
    "Clearance Level",
    options=["public", "employee", "manager", "admin"],
    index=0,
    help="Maximum document clearance level accessible to the caller.",
)

include_archived = st.sidebar.checkbox(
    "Include Archived Documents",
    value=False,
    help="When checked, bypasses default exclusion of deprecated/archived files.",
)

access_context = {
    "department": dept,
    "access_level": access_lvl,
    "include_archived": include_archived,
}

st.sidebar.markdown("---")
with st.sidebar.expander("📄 Document Ingestion", expanded=False):
    st.markdown(
        "<div style='font-size: 0.82rem; color: #334155; font-weight: 500; margin-bottom: 0.5rem;'>Upload a document (PDF, Markdown, TXT) to chunk, embed, and store into PostgreSQL:</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload File",
        type=["pdf", "md", "txt"],
        help="Supported formats: PDF (.pdf), Markdown (.md), Plain Text (.txt)",
        key="doc_uploader",
    )
    up_dept = st.selectbox("Document Department", ["public", "engineering", "finance", "hr"], index=0, key="up_dept")
    up_clearance = st.selectbox("Document Clearance", ["public", "employee", "manager", "admin"], index=0, key="up_clearance")
    extract_kg = st.checkbox("Extract Knowledge Graph (LLM)", value=False, help="Extract entities and relationships into PostgreSQL graph via Gemini.")

    if uploaded_file is not None:
        if st.button("Process & Index Document", type="primary", use_container_width=True):
            with st.spinner("Processing document chunks & generating 384-d vectors..."):
                try:
                    from ui.ingestion_helper import ingest_document_file
                    ingest_res = ingest_document_file(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        department=up_dept,
                        access_level=up_clearance,
                        extract_graph=extract_kg,
                    )
                    st.success(
                        f"**Ingested '{ingest_res['filename']}'!**\n\n"
                        f"- Document ID: `{ingest_res['document_id']}`\n"
                        f"- Chunks Stored: `{ingest_res['chunks_stored']}`\n"
                        f"- Vector Dimension: `{ingest_res['vector_dimension']}` (MiniLM)\n"
                        + (f"- Graph Triples: `{ingest_res['graph_stats'].get('relationships_inserted', 0)}` relationships\n" if extract_kg else "")
                    )
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")

st.sidebar.markdown("---")
with st.sidebar.expander("System Architecture", expanded=False):
    st.markdown(
        """
        **Pipeline Flow:**
        1. **Dual Ingestion**: PDF structure-aware chunks & entity/relationship graph.
        2. **Hybrid Search**: Dense embeddings (`all-MiniLM-L6-v2`) + BM25 + Reciprocal Rank Fusion.
        3. **Graph Traversal**: Seed extraction + bounded PostgreSQL BFS (depth 1-2).
        4. **Evidence Fusion**: Deduplicated union of vector & graph candidates.
        5. **Reranking**: `ms-marco-MiniLM-L-6-v2` cross-encoder scoring.
        6. **Grounded Generation**: Gemini 2.5 Flash with strict refusal guardrails.
        """
    )


# ---------------------------------------------------------------------------
# Main Layout
# ---------------------------------------------------------------------------

# 1. Header with live status
render_header(health_info, api_url)

if not health_info.get("ok", False):
    st.warning(
        f"**Backend Service Unreachable:** Unable to communicate with FastAPI at `{api_url}`. "
        "Please start the backend server:\n\n"
        "```powershell\nuvicorn app.api.main:app --host 0.0.0.0 --port 8000\n```"
    )

# 2. Query Input & Example Chips
query_text, submit_clicked = render_query_input()

# 3. Execution Handling
if submit_clicked:
    if not query_text.strip():
        st.error("Please enter a question or query before submitting.")
    elif not health_info.get("ok", False):
        st.error(
            f"Cannot execute query because the backend at `{api_url}` is offline. "
            "Start the FastAPI server first."
        )
    else:
        with st.spinner("Executing Dual-Engine Retrieval & Grounded Generation..."):
            res = api_client.submit_query(
                api_url=api_url,
                query=query_text,
                top_k=top_k,
                candidate_k=candidate_k,
                retrieval_mode=retrieval_mode,
                access_context=access_context,
            )

        if not res.get("ok", False):
            err_type = res.get("error_type", "Error")
            detail = res.get("detail", "An unexpected error occurred.")
            st.error(f"**Query Failed ({err_type}):** {detail}")
        else:
            st.session_state["last_response"] = res["data"]
            st.session_state["last_query_text"] = query_text

# 4. Results Rendering
if "last_response" in st.session_state:
    data = st.session_state["last_response"]
    citations = data.get("citations", [])
    diagnostics = data.get("diagnostics", {})

    st.markdown("---")

    # Tabs for organized, clean inspection
    tab_answer, tab_graph, tab_telemetry = st.tabs(
        [
            f"Grounded Answer & Citations ({len(citations)})",
            f"Knowledge Graph Evidence ({len(diagnostics.get('graph_relationships', []))})",
            "Pipeline Telemetry & Diagnostics",
        ]
    )

    with tab_answer:
        render_answer(data)
        render_citations(citations)

    with tab_graph:
        render_graph_evidence(diagnostics)

    with tab_telemetry:
        render_diagnostics(diagnostics)
