"""
Knowledge Graph Explorer View.

Provides an interactive visual network graph and entity/relationship inspector
for the PostgreSQL enterprise knowledge graph.
"""
from typing import Dict, Any, List, Optional
import html
import json
import streamlit as st
import streamlit.components.v1 as components
from app.db import get_connection


ENTITY_TYPE_COLORS = {
    "TECHNOLOGY": "#38bdf8",     # Sky blue
    "MODEL": "#a855f7",          # Purple
    "ORGANIZATION": "#f59e0b",   # Amber
    "PERSON": "#3b82f6",         # Blue
    "LOCATION": "#ec4899",       # Pink
    "DATASET": "#10b981",        # Emerald
    "METRIC": "#06b6d4",         # Cyan
    "CONCEPT": "#8b5cf6",        # Violet
    "ALGORITHM": "#6366f1",      # Indigo
    "REGULATION": "#ef4444",     # Red
    "PRODUCT": "#f97316",        # Orange
    "EVENT": "#14b8a6",          # Teal
}


def _get_graph_stats() -> Dict[str, Any]:
    """Retrieves high-level counts and breakdown from the database."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            total_entities = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM relationships")
            total_relationships = cur.fetchone()[0]
            cur.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"
            )
            type_counts = cur.fetchall()
        return {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "type_counts": type_counts,
        }
    except Exception as exc:
        return {"total_entities": 0, "total_relationships": 0, "type_counts": [], "error": str(exc)}


def _fetch_subgraph(
    search_query: str = "",
    selected_type: str = "ALL",
    limit: int = 50,
) -> Dict[str, Any]:
    """Queries entities and incident relationships from PostgreSQL."""
    nodes = []
    edges = []
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            # Query matching entities
            where_clauses = []
            params = []
            if search_query.strip():
                where_clauses.append("(canonical_name ILIKE %s OR display_name ILIKE %s)")
                params.extend([f"%{search_query.strip()}%", f"%{search_query.strip()}%"])
            if selected_type and selected_type != "ALL":
                where_clauses.append("entity_type = %s")
                params.append(selected_type)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            params.append(limit)

            cur.execute(
                f"""
                SELECT id, canonical_name, entity_type, display_name, metadata
                FROM entities
                {where_sql}
                ORDER BY id ASC
                LIMIT %s
                """,
                tuple(params),
            )
            entity_rows = cur.fetchall()
            entity_ids = [r[0] for r in entity_rows]

            for r in entity_rows:
                nodes.append({
                    "id": r[0],
                    "label": r[3] or r[1],
                    "title": f"<b>{html.escape(r[3] or r[1])}</b><br>Type: {r[2]}<br>ID: {r[0]}",
                    "group": r[2],
                    "color": ENTITY_TYPE_COLORS.get(r[2], "#94a3b8"),
                })

            if entity_ids:
                cur.execute(
                    """
                    SELECT r.id, r.source_entity_id, r.target_entity_id, r.relationship_type,
                           r.chunk_id, r.document_id,
                           e1.display_name AS src_name, e2.display_name AS tgt_name
                    FROM relationships r
                    JOIN entities e1 ON r.source_entity_id = e1.id
                    JOIN entities e2 ON r.target_entity_id = e2.id
                    WHERE r.source_entity_id = ANY(%s) OR r.target_entity_id = ANY(%s)
                    LIMIT 100
                    """,
                    (entity_ids, entity_ids),
                )
                rel_rows = cur.fetchall()
                seen_edges = set()
                node_ids = {n["id"] for n in nodes}

                for r in rel_rows:
                    edge_key = (r[1], r[2], r[3])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        # Add missing neighbor nodes if incident
                        if r[1] not in node_ids:
                            node_ids.add(r[1])
                            nodes.append({
                                "id": r[1],
                                "label": r[6] or f"Entity {r[1]}",
                                "title": f"<b>{html.escape(r[6] or '')}</b><br>ID: {r[1]}",
                                "group": "NEIGHBOR",
                                "color": "#64748b",
                            })
                        if r[2] not in node_ids:
                            node_ids.add(r[2])
                            nodes.append({
                                "id": r[2],
                                "label": r[7] or f"Entity {r[2]}",
                                "title": f"<b>{html.escape(r[7] or '')}</b><br>ID: {r[2]}",
                                "group": "NEIGHBOR",
                                "color": "#64748b",
                            })

                        edges.append({
                            "from": r[1],
                            "to": r[2],
                            "label": r[3],
                            "title": f"Rel: {r[3]}<br>Chunk: {r[4]}<br>Doc: {r[5]}",
                            "arrows": "to",
                        })

        return {"nodes": nodes, "edges": edges, "error": None}
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}


def render_visjs_graph(nodes: List[Dict], edges: List[Dict], height: int = 580) -> None:
    """Renders a responsive interactive Vis.js graph inside an iframe component."""
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background-color: #0b0f19;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            #network {{
                width: 100%;
                height: {height}px;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }}
            #legend {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(4px);
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                color: #e2e8f0;
                font-size: 11px;
                z-index: 10;
                pointer-events: none;
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 10px;
            }}
            .legend-dot {{
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                margin-right: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="legend">
            <span class="legend-item"><span class="legend-dot" style="background: #38bdf8;"></span>TECH</span>
            <span class="legend-item"><span class="legend-dot" style="background: #a855f7;"></span>MODEL</span>
            <span class="legend-item"><span class="legend-dot" style="background: #f59e0b;"></span>ORG</span>
            <span class="legend-item"><span class="legend-dot" style="background: #3b82f6;"></span>PERSON</span>
            <span class="legend-item"><span class="legend-dot" style="background: #10b981;"></span>DATASET</span>
            <span class="legend-item"><span class="legend-dot" style="background: #ec4899;"></span>LOC</span>
        </div>
        <div id="network"></div>
        <script type="text/javascript">
            var rawNodes = {nodes_json};
            var rawEdges = {edges_json};

            var nodes = new vis.DataSet(rawNodes.map(function(n) {{
                return {{
                    id: n.id,
                    label: n.label,
                    title: n.title,
                    color: {{
                        background: n.color || "#38bdf8",
                        border: "#ffffff",
                        highlight: {{ background: "#38bdf8", border: "#ffffff" }}
                    }},
                    font: {{ color: "#f8fafc", size: 12, face: "sans-serif" }},
                    shape: "dot",
                    size: 16,
                    shadow: {{ enabled: true, color: "rgba(0,0,0,0.5)", size: 8, x: 2, y: 2 }}
                }};
            }}));

            var edges = new vis.DataSet(rawEdges.map(function(e) {{
                return {{
                    from: e.from,
                    to: e.to,
                    label: e.label,
                    title: e.title,
                    arrows: "to",
                    color: {{ color: "#64748b", highlight: "#38bdf8" }},
                    font: {{ color: "#94a3b8", size: 10, align: "middle", background: "#0b0f19" }},
                    smooth: {{ type: "continuous" }},
                    width: 1.5
                }};
            }}));

            var container = document.getElementById('network');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                physics: {{
                    stabilization: {{ iterations: 120 }},
                    barnesHut: {{
                        gravitationalConstant: -3500,
                        springConstant: 0.04,
                        springLength: 120
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    navigationButtons: true,
                    keyboard: true
                }}
            }};

            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height + 20)


def render_graph_view() -> None:
    """Main render function for the Knowledge Graph Explorer tab."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-bottom: 0.35rem;">
                🕸️ Enterprise Knowledge Graph Explorer
            </h1>
            <p style="color: #475569; font-size: 0.95rem;">
                Explore discovered entities, semantic predicates, and multi-hop relationship chains stored in PostgreSQL.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = _get_graph_stats()

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entities", f"{stats['total_entities']:,}")
    with col2:
        st.metric("Total Relationships", f"{stats['total_relationships']:,}")
    with col3:
        st.metric("Entity Types", len(stats["type_counts"]))
    with col4:
        st.metric("Traversal Engine", "Depth-2 Multi-Hop")

    st.markdown("---")

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1])
    with f_col1:
        search_kw = st.text_input(
            "Search Entity Name",
            value="",
            placeholder="e.g. Lumivastin-4, AuroraMesh, Transformer, Elena Rostova...",
        )
    with f_col2:
        all_types = ["ALL"] + [t[0] for t in stats["type_counts"]]
        selected_type = st.selectbox("Filter Entity Type", options=all_types, index=0)
    with f_col3:
        node_limit = st.slider("Node Limit", min_value=10, max_value=120, value=40, step=10)

    # Fetch and Render Graph
    subgraph = _fetch_subgraph(search_query=search_kw, selected_type=selected_type, limit=node_limit)

    if subgraph["error"]:
        st.error(f"Error querying knowledge graph: {subgraph['error']}")
        return

    nodes = subgraph["nodes"]
    edges = subgraph["edges"]

    if not nodes:
        st.info("No entities match the current search or filter criteria.")
        return

    st.markdown(
        f"<div style='font-size: 0.9rem; color: #475569; margin-bottom: 0.5rem; font-weight: 600;'>"
        f"Rendering <b>{len(nodes)}</b> entities and <b>{len(edges)}</b> traversed relationships "
        f"(interactive zoom/pan/drag enabled):</div>",
        unsafe_allow_html=True,
    )

    render_visjs_graph(nodes=nodes, edges=edges, height=560)

    # Tabbed Detail Tables
    t1, t2 = st.tabs(["📋 Rendered Entities", "🔗 Rendered Relationships"])
    with t1:
        ent_display = [
            {"Entity ID": n["id"], "Label": n["label"], "Type": n.get("group", "UNKNOWN")}
            for n in nodes
        ]
        st.dataframe(ent_display, use_container_width=True, hide_index=True)
    with t2:
        rel_display = [
            {"Source ID": e["from"], "Relationship": e["label"], "Target ID": e["to"]}
            for e in edges
        ]
        st.dataframe(rel_display, use_container_width=True, hide_index=True)
