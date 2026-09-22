"""
Diagnostics View Component.

Displays pipeline telemetry, candidate pooling, latency breakdown across stages,
token usage, and raw diagnostic metadata.
"""
from typing import Dict, Any
import streamlit as st


def render_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """
    Render pipeline diagnostics and operational telemetry.

    Args:
        diagnostics: The diagnostics dictionary from QueryResponse
    """
    if not diagnostics:
        st.info("No operational diagnostics returned.")
        return

    st.markdown(
        "<div style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;'>Pipeline Telemetry & Diagnostics</div>",
        unsafe_allow_html=True,
    )

    # 1. Candidate Counts Metric Tiles
    v_cands = diagnostics.get("vector_candidates_count", 0)
    g_cands = diagnostics.get("graph_candidates_count", 0)
    comb_cands = diagnostics.get("combined_unique_count", 0)
    reranked = diagnostics.get("reranked_count", 0)

    st.markdown(
        "<div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem;'>CANDIDATE POOLING & DEDUPLICATION:</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #38bdf8;">{v_cands}</div>
                <div class="metric-lbl">Vector / BM25</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #c084fc;">{g_cands}</div>
                <div class="metric-lbl">Graph Traversal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #34d399;">{comb_cands}</div>
                <div class="metric-lbl">Unique Combined</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #818cf8;">{reranked}</div>
                <div class="metric-lbl">Reranked Output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

    # 2. Latency Breakdown
    v_lat = diagnostics.get("vector_latency_ms")
    g_lat = diagnostics.get("graph_latency_ms")
    r_lat = diagnostics.get("rerank_latency_ms")
    tot_lat = diagnostics.get("total_latency_ms")
    gen_time_s = diagnostics.get("generation_time_sec")
    gen_lat = gen_time_s * 1000.0 if gen_time_s is not None else None

    st.markdown(
        "<div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem;'>EXECUTION LATENCY BREAKDOWN (MS):</div>",
        unsafe_allow_html=True,
    )
    l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns(5)
    with l_col1:
        st.metric("Vector Stage", f"{v_lat:.1f} ms" if v_lat is not None else "N/A")
    with l_col2:
        st.metric("Graph Stage", f"{g_lat:.1f} ms" if g_lat is not None else "N/A")
    with l_col3:
        st.metric("Reranker Stage", f"{r_lat:.1f} ms" if r_lat is not None else "N/A")
    with l_col4:
        st.metric("LLM Gen", f"{gen_lat:.0f} ms" if gen_lat is not None else "N/A")
    with l_col5:
        st.metric("Total Latency", f"{tot_lat:.1f} ms" if tot_lat is not None else "N/A")

    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)

    # 3. Token & Budget Details
    p_tok = diagnostics.get("prompt_tokens")
    r_tok = diagnostics.get("response_tokens")
    budget_used = diagnostics.get("context_budget_used_tokens") or diagnostics.get("context_chars_used")

    if p_tok is not None or budget_used is not None:
        st.markdown(
            "<div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem;'>CONTEXT & TOKEN ALLOCATION:</div>",
            unsafe_allow_html=True,
        )
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("Prompt Tokens", p_tok if p_tok is not None else "N/A")
        with t_col2:
            st.metric("Response Tokens", r_tok if r_tok is not None else "N/A")
        with t_col3:
            st.metric("Context Budget Used", f"{budget_used}" if budget_used is not None else "N/A")

    # 4. Raw JSON Inspector
    with st.expander("Inspect Raw Diagnostics JSON", expanded=False):
        st.json(diagnostics)
