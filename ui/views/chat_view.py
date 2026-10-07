"""
Chat View Component.

Conversational chatbot interface for Enterprise Graph RAG.
Features:
- Chat bubbles for user questions and assistant answers.
- Formatted source attribution tags ([SOURCE 1]).
- Expandable citations with full passage inspection and provenance badges.
- Expandable knowledge graph traversal paths.
- Pipeline diagnostics & latency metrics.
- Preset example question chips for quick testing.
"""
from typing import Dict, Any, List
import html
import streamlit as st

from ui import api_client
from ui.components.answer_view import render_answer
from ui.components.citations_view import render_citations
from ui.components.graph_evidence_view import render_graph_evidence
from ui.components.diagnostics_view import render_diagnostics

SAMPLE_QUERIES = [
    {
        "label": "Chunking Metadata",
        "query": "What metadata is preserved during the chunking phase?",
    },
    {
        "label": "Chunking Precision",
        "query": "Why does naive fixed-window chunking degrade retrieval precision according to platform architecture?",
    },
    {
        "label": "AI Governance",
        "query": "What controls are required before any document is vectorized or indexed?",
    },
    {
        "label": "RBAC & Security",
        "query": "What security controls and access levels gate underlying vector stores?",
    },
    {
        "label": "Negative Control",
        "query": "What is the capital of Mars?",
    },
]


def render_chat_view(
    api_url: str,
    health_info: Dict[str, Any],
    retrieval_mode: str,
    top_k: int,
    candidate_k: int,
    access_context: Dict[str, Any],
) -> None:
    """
    Renders the conversational Chatbot interface.
    """
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # 1. Header Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em;">
                Enterprise Knowledge Assistant
            </div>
            <div style="font-size: 0.92rem; color: #475569; margin-top: 0.2rem;">
                Dual-engine conversational retrieval powered by pgvector, BM25, and PostgreSQL Knowledge Graph.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Show Prompt Chips if chat is empty
    if not st.session_state["chat_messages"]:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">
                    👋 Welcome! Ask any question across your enterprise documents.
                </div>
                <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                    The assistant searches through vector embeddings and traverses the knowledge graph to provide strictly grounded answers with verifiable source citations and RBAC clearance filtering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.8rem; font-weight: 700; color: #334155; margin-bottom: 0.4rem;'>EXAMPLE QUESTIONS:</div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(len(SAMPLE_QUERIES))
        for idx, item in enumerate(SAMPLE_QUERIES):
            with chip_cols[idx]:
                if st.button(item["label"], key=f"chip_btn_{idx}", use_container_width=True):
                    st.session_state["pending_query"] = item["query"]
                    st.rerun()

    # 3. Render Historical Chat Messages
    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{msg['content']}**")
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                resp_data = msg.get("response_data", {})
                citations = resp_data.get("citations", [])
                diagnostics = resp_data.get("diagnostics", {})
                relationships = diagnostics.get("graph_relationships", [])

                # Render grounded answer card
                render_answer(resp_data)

                # Expandable citations
                if citations:
                    with st.expander(f"📚 View Authoritative Citations ({len(citations)})", expanded=False):
                        render_citations(citations)

                # Expandable knowledge graph evidence
                if relationships:
                    with st.expander(f"🕸️ Knowledge Graph Traversal ({len(relationships)} edges)", expanded=False):
                        render_graph_evidence(diagnostics)

                # Expandable diagnostics
                if diagnostics:
                    with st.expander("⏱️ Pipeline Diagnostics & Latency Breakdown", expanded=False):
                        render_diagnostics(diagnostics)

    # 4. Handle Pending Query from Prompt Chips
    auto_query = None
    if "pending_query" in st.session_state and st.session_state["pending_query"]:
        auto_query = st.session_state.pop("pending_query")

    # 5. Chat Input Field
    chat_prompt = st.chat_input("Ask a question about your enterprise documents...")
    effective_query = auto_query or chat_prompt

    if effective_query:
        # Check backend health
        if not health_info.get("ok", False):
            st.error(f"Cannot execute query because the backend at `{api_url}` is offline.")
            return

        # Append user message
        st.session_state["chat_messages"].append({
            "role": "user",
            "content": effective_query,
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(f"**{effective_query}**")

        # Execute query with spinner in assistant message
        with st.chat_message("assistant"):
            with st.spinner("Searching vectors, traversing knowledge graph & generating grounded answer..."):
                res = api_client.submit_query(
                    api_url=api_url,
                    query=effective_query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    retrieval_mode=retrieval_mode,
                    access_context=access_context,
                )

            if not res.get("ok", False):
                err_msg = res.get("detail", "An unexpected error occurred.")
                st.error(f"Query Failed: {err_msg}")
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "response_data": {"answer": f"Error: {err_msg}", "citations": []},
                })
            else:
                resp_data = res["data"]
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "response_data": resp_data,
                })
                st.rerun()
