"""
Header Component.

Renders the application title, architecture subtitle, and dynamic backend status indicator.
"""
import textwrap
import html
from typing import Dict, Any
import streamlit as st


def render_header(health_data: Dict[str, Any], api_url: str) -> None:
    """
    Render top title bar and backend connection pill.

    Args:
        health_data: Result dictionary from api_client.check_health()
        api_url: The current active API base URL
    """
    is_online = health_data.get("ok", False)
    db_state = health_data.get("database", "disconnected")
    version = health_data.get("version", "1.0.0")

    if is_online and db_state == "connected":
        status_html = (
            '<div class="status-pill status-online">'
            '<span class="status-dot dot-online"></span>'
            '<span>API Online &bull; DB Connected</span>'
            '</div>'
        )
    elif is_online:
        status_html = (
            '<div class="status-pill" style="background-color: #fef3c7; border: 1px solid #f59e0b; color: #92400e;">'
            '<span class="status-dot" style="background-color: #d97706; box-shadow: 0 0 6px #d97706;"></span>'
            f'<span>API Online &bull; DB {html.escape(db_state.upper())}</span>'
            '</div>'
        )
    else:
        status_html = (
            '<div class="status-pill status-offline">'
            '<span class="status-dot dot-offline"></span>'
            '<span>API Offline</span>'
            '</div>'
        )

    col_title, col_status = st.columns([3.8, 1.4])
    with col_title:
        title_html = textwrap.dedent("""
        <div class="header-container">
            <h1 class="header-title">Enterprise Graph RAG</h1>
            <p class="header-subtitle">
                Dual-Engine Retrieval: Dense Vector + BM25 + PostgreSQL Knowledge Graph with Provenance
            </p>
        </div>
        """).strip()
        st.markdown(title_html, unsafe_allow_html=True)

    with col_status:
        status_box_html = textwrap.dedent(f"""
        <div style="padding-top: 1.25rem; text-align: right;">
        {status_html}
        <div style="font-size: 0.75rem; color: #334155; margin-top: 0.35rem; font-weight: 500;">
        v{html.escape(str(version))} &bull; <code>{html.escape(str(api_url))}</code>
        </div>
        </div>
        """).strip()
        st.markdown(status_box_html, unsafe_allow_html=True)
