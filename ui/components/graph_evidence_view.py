"""
Knowledge Graph Evidence Component.

Visualizes recognized query seed entities, traversed relationship triples, and
chunk/document provenance from the PostgreSQL knowledge graph.
"""
from typing import Dict, Any, List
import html
import streamlit as st


def render_graph_evidence(diagnostics: Dict[str, Any]) -> None:
    """
    Render graph traversal paths and relationship provenance.

    Args:
        diagnostics: The diagnostics dictionary from QueryResponse
    """
    seed_entities: List[str] = diagnostics.get("graph_seed_entities", [])
    relationships: List[Dict[str, Any]] = diagnostics.get("graph_relationships", [])
    graph_count = diagnostics.get("graph_candidates_count", 0)

    st.markdown(
        "<div style='font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem;'>Knowledge Graph Traversal</div>",
        unsafe_allow_html=True,
    )

    # 1. Seed Entities Section
    if seed_entities:
        st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.4rem;'>RECOGNIZED QUERY SEED ENTITIES:</div>", unsafe_allow_html=True)
        seeds_html = " ".join(
            f'<span class="graph-entity-source" style="margin-right: 0.4rem; display: inline-block; margin-bottom: 0.35rem;">{html.escape(s)}</span>'
            for s in seed_entities
        )
        st.markdown(f"<div style='margin-bottom: 1.25rem;'>{seeds_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color: #94a3b8; font-size: 0.88rem; margin-bottom: 1rem;'>No specific entity seeds identified in query text.</div>",
            unsafe_allow_html=True,
        )

    # 2. Relationship Triples Section
    if relationships:
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.5rem;'>TRAVERSED GRAPH EDGES ({len(relationships)}):</div>",
            unsafe_allow_html=True,
        )

        for rel in relationships:
            src = rel.get("source", "Unknown")
            rel_type = rel.get("type") or rel.get("relationship_type", "RELATED_TO")
            tgt = rel.get("target", "Unknown")
            chunk_id = rel.get("chunk_id", "N/A")
            page_num = rel.get("page_number")
            doc_id = rel.get("document_id", "N/A")

            page_display = f"Page {page_num}" if page_num else "Page N/A"

            card_html = f"""
            <div class="graph-rel-card">
                <div class="graph-triple">
                    <span class="graph-entity-source">{html.escape(src)}</span>
                    <span class="graph-predicate">&mdash;&mdash;[{html.escape(rel_type)}]&mdash;&mdash;&gt;</span>
                    <span class="graph-entity-target">{html.escape(tgt)}</span>
                </div>
                <div class="graph-provenance">
                    <span>Provenance: <b>Chunk {html.escape(str(chunk_id))}</b> ({page_display})</span>
                    <span>Doc ID: <code>{html.escape(str(doc_id))}</code></span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    elif graph_count > 0:
        st.info(f"Retrieved {graph_count} chunk candidates via entity graph mapping.")
    else:
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.15); border-radius: 6px; padding: 1rem; color: #94a3b8; font-size: 0.88rem;">
                No knowledge graph traversal edges were active for this query. Results were fulfilled exclusively via Dense Vector and BM25 lexical channels.
            </div>
            """,
            unsafe_allow_html=True,
        )
