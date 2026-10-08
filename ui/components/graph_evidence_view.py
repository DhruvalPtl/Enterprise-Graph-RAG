"""
Knowledge Graph Evidence Component.

Visualizes recognized query seed entities, traversed relationship triples, and
chunk/document provenance from the PostgreSQL knowledge graph.
"""
from typing import Dict, Any, List
import html
import textwrap
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
        "<div style='font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem;'>Knowledge Graph Traversal</div>",
        unsafe_allow_html=True,
    )

    # 1. Seed Entities Section
    if seed_entities:
        st.markdown("<div style='font-size: 0.85rem; color: #334155; font-weight: 700; margin-bottom: 0.4rem;'>RECOGNIZED QUERY SEED ENTITIES:</div>", unsafe_allow_html=True)
        seeds_html = " ".join(
            f'<span class="graph-entity-source" style="margin-right: 0.4rem; display: inline-block; margin-bottom: 0.35rem;">{html.escape(s)}</span>'
            for s in seed_entities
        )
        st.markdown(f"<div style='margin-bottom: 1.25rem;'>{seeds_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color: #475569; font-size: 0.88rem; font-weight: 500; margin-bottom: 1rem;'>No specific entity seeds identified in query text.</div>",
            unsafe_allow_html=True,
        )

    # 2. Relationship Triples Section
    if relationships:
        st.markdown(
            f"<div style='font-size: 0.85rem; color: #334155; font-weight: 700; margin-bottom: 0.5rem;'>TRAVERSED GRAPH EDGES ({len(relationships)}):</div>",
            unsafe_allow_html=True,
        )

        # Render interactive visual sub-graph canvas
        try:
            import json
            import streamlit.components.v1 as components

            node_map = {}
            vis_nodes = []
            vis_edges = []
            seed_set = {s.strip().lower() for s in seed_entities}

            for idx, r in enumerate(relationships):
                s = str(r.get("source", "Unknown")).strip()
                t = str(r.get("target", "Unknown")).strip()
                rtype = str(r.get("type") or r.get("relationship_type", "RELATED_TO")).strip()

                if s and s not in node_map:
                    node_map[s] = len(node_map) + 1
                    is_seed = s.lower() in seed_set
                    vis_nodes.append({
                        "id": node_map[s],
                        "label": s,
                        "color": "#f59e0b" if is_seed else "#38bdf8",
                        "size": 18 if is_seed else 14,
                        "font": {"color": "#ffffff", "size": 12, "face": "sans-serif"},
                    })

                if t and t not in node_map:
                    node_map[t] = len(node_map) + 1
                    is_seed = t.lower() in seed_set
                    vis_nodes.append({
                        "id": node_map[t],
                        "label": t,
                        "color": "#f59e0b" if is_seed else "#a855f7",
                        "size": 18 if is_seed else 14,
                        "font": {"color": "#ffffff", "size": 12, "face": "sans-serif"},
                    })

                if s in node_map and t in node_map:
                    vis_edges.append({
                        "from": node_map[s],
                        "to": node_map[t],
                        "label": rtype,
                        "arrows": "to",
                        "color": {"color": "#64748b", "highlight": "#38bdf8"},
                        "font": {"color": "#94a3b8", "size": 10, "background": "#0b0f19"},
                    })

            if vis_nodes:
                canvas_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                    <style>
                        html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0b0f19; }}
                        #subnetwork {{ width:100%; height:260px; border:1px solid #1e293b; border-radius:6px; }}
                    </style>
                </head>
                <body>
                    <div id="subnetwork"></div>
                    <script>
                        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
                        var edges = new vis.DataSet({json.dumps(vis_edges)});
                        var container = document.getElementById('subnetwork');
                        var data = {{ nodes: nodes, edges: edges }};
                        var options = {{
                            physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -2500, springLength: 100 }} }},
                            interaction: {{ hover: true, zoomView: true, dragView: true }}
                        }};
                        var network = new vis.Network(container, data, options);
                    </script>
                </body>
                </html>
                """
                components.html(canvas_html, height=275)
        except Exception:
            pass

        for rel in relationships:
            src = rel.get("source", "Unknown")
            rel_type = rel.get("type") or rel.get("relationship_type", "RELATED_TO")
            tgt = rel.get("target", "Unknown")
            chunk_id = rel.get("chunk_id", "N/A")
            page_num = rel.get("page_number")
            doc_id = rel.get("document_id", "N/A")

            page_display = f"Page {page_num}" if page_num else "Page N/A"

            card_html = textwrap.dedent(f"""
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
            """).strip()
            st.markdown(card_html, unsafe_allow_html=True)
    elif graph_count > 0:
        st.info(f"Retrieved {graph_count} chunk candidates via entity graph mapping.")
    else:
        empty_html = textwrap.dedent("""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px; padding: 1rem; color: #334155; font-size: 0.9rem; font-weight: 500;">
            No knowledge graph traversal edges were active for this query. Results were fulfilled exclusively via Dense Vector and BM25 lexical channels.
        </div>
        """).strip()
        st.markdown(empty_html, unsafe_allow_html=True)
