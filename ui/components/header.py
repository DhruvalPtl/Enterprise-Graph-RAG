"""
Header Component.

Renders the application title, architecture subtitle, and dynamic backend status indicator.
"""
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
        status_html = """
        <div class="status-pill status-online">
            <span class="status-dot dot-online"></span>
            <span>API Online &bull; DB Connected</span>
        </div>
        """
    elif is_online:
        status_html = f"""
        <div class="status-pill" style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); color: #fbbf24;">
            <span class="status-dot" style="background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b;"></span>
            <span>API Online &bull; DB {db_state.upper()}</span>
        </div>
        """
    else:
        status_html = """
        <div class="status-pill status-offline">
            <span class="status-dot dot-offline"></span>
            <span>API Offline</span>
        </div>
        """

    col_title, col_status = st.columns([4, 1.2])
    with col_title:
        st.markdown(
            """
            <div class="header-container">
                <h1 class="header-title">Enterprise Graph RAG</h1>
                <p class="header-subtitle">
                    Dual-Engine Retrieval: Dense Vector + BM25 + PostgreSQL Knowledge Graph with Provenance
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        st.markdown(
            f"""
            <div style="padding-top: 1.5rem; text-align: right;">
                {status_html}
                <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.35rem;">
                    v{version} &bull; <code>{api_url}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
