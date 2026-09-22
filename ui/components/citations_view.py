"""
Citations View Component.

Renders authoritative source evidence cards with document, page, reranker score,
and dual-engine retrieval provenance (Vector / Graph / Both).
"""
from typing import List, Dict, Any
import html
import streamlit as st


def _render_provenance_badge(retrieved_by: Any) -> str:
    """Generate HTML badge based on retrieval origin."""
    if not retrieved_by:
        return '<span class="provenance-badge badge-vector">Dense / BM25</span>'

    sources = set(s.lower() for s in retrieved_by) if isinstance(retrieved_by, list) else {str(retrieved_by).lower()}

    if "vector" in sources and "graph" in sources:
        return '<span class="provenance-badge badge-both">&bull; HYBRID VERIFIED (Vector + Graph)</span>'
    elif "graph" in sources:
        return '<span class="provenance-badge badge-graph">&bull; GRAPH TRAVERSAL</span>'
    elif "vector" in sources:
        return '<span class="provenance-badge badge-vector">&bull; DENSE / BM25</span>'
    else:
        return f'<span class="provenance-badge badge-vector">&bull; {html.escape(str(retrieved_by))}</span>'


def render_citations(citations: List[Dict[str, Any]]) -> None:
    """
    Render all source citations in structured, inspectable cards.

    Args:
        citations: List of CitationResponse dictionaries
    """
    if not citations:
        st.info("No source citations were referenced for this query.")
        return

    st.markdown(
        f"<div style='font-size: 1.1rem; font-weight: 700; margin-top: 1rem; margin-bottom: 0.75rem;'>Authoritative Citations ({len(citations)})</div>",
        unsafe_allow_html=True,
    )

    for c in citations:
        source_id = c.get("source_id", 1)
        filename = c.get("filename", "Unknown Document")
        page_num = c.get("page_number")
        section = c.get("section")
        reranker_score = c.get("reranker_score")
        rrf_score = c.get("rrf_score")
        retrieved_by = c.get("retrieved_by")
        content = c.get("content") or ""

        page_str = f"Page {page_num}" if page_num else "Page N/A"
        section_str = f" &bull; <i>{html.escape(section)}</i>" if section else ""
        badge_html = _render_provenance_badge(retrieved_by)

        score_parts = []
        if reranker_score is not None:
            score_parts.append(f"Reranker: <b>{reranker_score:+.3f}</b>")
        if rrf_score is not None:
            score_parts.append(f"RRF: <b>{rrf_score:.4f}</b>")
        score_str = " | ".join(score_parts)

        card_html = f"""
        <div class="citation-card">
            <div class="citation-header">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span class="source-tag">SOURCE {source_id}</span>
                    <span style="font-weight: 600; color: #f8fafc; font-size: 0.92rem;">{html.escape(filename)}</span>
                </div>
                <div>
                    {badge_html}
                </div>
            </div>
            <div class="citation-meta">
                <span>{page_str}{section_str}</span>
                <span style="float: right; color: #94a3b8; font-size: 0.8rem;">{score_str}</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Expandable passage text for full evidence transparency
        if content:
            with st.expander(f"Inspect Evidence Passage [SOURCE {source_id}]", expanded=False):
                st.markdown(
                    f"<div class='citation-passage'>{html.escape(content)}</div>",
                    unsafe_allow_html=True,
                )
                chunk_id = c.get("chunk_id")
                doc_id = c.get("document_id")
                st.caption(f"Chunk ID: `{chunk_id}` | Doc ID: `{doc_id}`")
