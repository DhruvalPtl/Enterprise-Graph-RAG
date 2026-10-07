"""
Query Input Component.

Provides a search bar, preset example query chips grounded in the enterprise corpus,
and submission trigger.
"""
from typing import Tuple
import streamlit as st

SAMPLE_QUERIES = [
    {
        "label": "AI Governance",
        "query": "What controls are required before a document can be vectorized according to the AI governance policy?",
        "desc": "Tests parsing integrity, metadata attachment, and PII redaction standards",
    },
    {
        "label": "Architecture",
        "query": "What are the core architectural layers and data ingestion stages of the enterprise platform?",
        "desc": "Tests multi-page PDF architectural extraction and citation attribution",
    },
    {
        "label": "Support FAQ",
        "query": "What is the standard support policy and resolution procedure for platform access issues?",
        "desc": "Tests plain text FAQ extraction and grounded synthesis",
    },
    {
        "label": "RBAC & Security",
        "query": "What security controls and access levels gate underlying vector stores?",
        "desc": "Tests department and clearance level security verification",
    },
    {
        "label": "Negative Control",
        "query": "What is the capital of Mars?",
        "desc": "Tests grounded refusal and zero-hallucination guardrails",
    },
]


def render_query_input() -> Tuple[str, bool]:
    """
    Render search query input, sample query chips, and submit button.

    Returns:
        Tuple of (entered_query: str, submit_button_clicked: bool)
    """
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = SAMPLE_QUERIES[0]["query"]

    # Sample query chips
    st.markdown(
        "<div style='font-size: 0.82rem; color: #334155; font-weight: 700; margin-bottom: 0.35rem;'>EXAMPLE CORPUS QUERIES:</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(SAMPLE_QUERIES))
    for idx, sample in enumerate(SAMPLE_QUERIES):
        with cols[idx]:
            if st.button(
                sample["label"],
                key=f"sample_btn_{idx}",
                help=f"{sample['query']}\n({sample['desc']})",
                use_container_width=True,
            ):
                st.session_state["current_query"] = sample["query"]
                st.rerun()

    query_val = st.text_area(
        label="Enter your query or research question:",
        value=st.session_state["current_query"],
        height=90,
        placeholder="e.g. What organization developed Gemini Ultra and what benchmark was it evaluated on?",
        help="Type any technical or multi-hop question. The platform will query both vector embeddings and the entity knowledge graph.",
    )

    # Sync back to session state
    st.session_state["current_query"] = query_val

    col_sub, col_info = st.columns([1.5, 4])
    with col_sub:
        submit = st.button(
            "Execute Retrieval & Answer",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.markdown(
            "<div style='font-size: 0.8rem; color: #475569; font-weight: 500; padding-top: 0.5rem;'>Press to invoke dense vector, lexical BM25, and PostgreSQL graph traversal pipelines.</div>",
            unsafe_allow_html=True,
        )

    return query_val, submit
