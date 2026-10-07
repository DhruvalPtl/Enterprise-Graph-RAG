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

import logging
try:
    from streamlit.logger import get_logger as st_get_logger
    st_get_logger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
except Exception:
    pass
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from ui import api_client
from ui.components.header import render_header
from ui.views.chat_view import render_chat_view
from ui.views.files_view import render_files_view
from ui.views.ingest_view import render_ingest_view

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
# Sidebar Navigation & Configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem;">Navigation</div>',
    unsafe_allow_html=True,
)

current_page = st.sidebar.radio(
    "Select View",
    options=["💬 Chat Assistant", "📁 Knowledge Base Files", "📤 Upload & Ingest"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">System Configuration</div>',
    unsafe_allow_html=True,
)

default_api_url = os.getenv("RAG_API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input(
    "FastAPI Service URL",
    value=default_api_url,
    help="Target FastAPI endpoint hosting POST /query and GET /health.",
)

api_key_val = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Google Gemini API key for answer synthesis and graph extraction.",
)
if api_key_val and api_key_val != os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = api_key_val

# Immediate backend health check
health_info = api_client.check_health(api_url)

# Retrieval Engine Settings
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

# RBAC Settings
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

# Quick actions in sidebar
if current_page == "💬 Chat Assistant":
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["chat_messages"] = []
        st.rerun()

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
# Main Layout & Page Routing
# ---------------------------------------------------------------------------

# Top Header with live status
render_header(health_info, api_url)

if not health_info.get("ok", False):
    st.warning(
        f"**Backend Service Unreachable:** Unable to communicate with FastAPI at `{api_url}`. "
        "Please verify your backend server is running:\n\n"
        "```powershell\nuvicorn app.api.main:app --host 0.0.0.0 --port 8000\n```"
    )

st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

# Page Routing
if current_page == "💬 Chat Assistant":
    render_chat_view(
        api_url=api_url,
        health_info=health_info,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        candidate_k=candidate_k,
        access_context=access_context,
    )
elif current_page == "📁 Knowledge Base Files":
    render_files_view()
elif current_page == "📤 Upload & Ingest":
    render_ingest_view()
