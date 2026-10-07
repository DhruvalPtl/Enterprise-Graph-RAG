"""
Query Input Component.

Provides a search bar, preset example query chips grounded in the enterprise corpus,
and submission trigger.
"""
from typing import Tuple
import streamlit as st

SAMPLE_QUERIES = [
    {
        "label": "Multi-Hop Graph",
        "query": "What organization developed Gemini Ultra and what benchmark was it evaluated on?",
        "desc": "Google DeepMind / Gemini -> MMLU benchmark evaluation",
    },
    {
        "label": "Policy Timeline",
        "query": "What happened on July 25, 2023?",
        "desc": "Senate hearing on AI oversight / Outbound Investment Act",
    },
    {
        "label": "Semantic RAG",
        "query": "What is the Transformer architecture, and what are its main components?",
        "desc": "Self-attention, Multi-Head Attention, Encoder-Decoder",
    },
    {
        "label": "Entity Commitments",
        "query": "What voluntary commitments did private AI labs sign in July 2023?",
        "desc": "White House voluntary commitments for safe AI",
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
