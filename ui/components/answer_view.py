"""
Answer View Component.

Renders grounded LLM generation, refuses gracefully when no evidence is found,
and displays generation metadata.
"""
from typing import Dict, Any, List
import re
import html
import textwrap
import streamlit as st

REFUSAL_PHRASES = [
    "i cannot find information",
    "does not contain information",
    "cannot answer",
    "insufficient evidence",
    "no evidence provided",
    "not mentioned in the provided",
    "no mention of",
]


def _is_refusal(answer: str, citations: List[Dict[str, Any]]) -> bool:
    """Check if the answer is a grounded refusal or if no citations exist."""
    if not citations:
        return True
    lower = answer.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


def _format_source_tags(text: str) -> str:
    """Format [SOURCE X] occurrences with stylish badge markup."""
    def repl(m: re.Match) -> str:
        src_num = m.group(1)
        return f'<span class="source-tag" style="padding: 0.1rem 0.4rem; font-size: 0.75rem; margin: 0 0.2rem;">SOURCE {src_num}</span>'
    return re.sub(r"\[SOURCE\s+(\d+)\]", repl, text)


def render_answer(response_data: Dict[str, Any]) -> None:
    """
    Render answer container with refusal detection and metadata.

    Args:
        response_data: QueryResponse dictionary from API
    """
    answer = response_data.get("answer", "")
    citations = response_data.get("citations", [])
    model_name = response_data.get("model_name", "Gemini")
    diagnostics = response_data.get("diagnostics", {})

    is_refusal = _is_refusal(answer, citations)

    if is_refusal:
        refusal_html = textwrap.dedent(f"""
        <div class="refusal-box">
            <div class="refusal-title">&bull; GROUNDED REFUSAL &bull; Zero-Hallucination Guardrail Active</div>
            <div style="font-size: 1rem; color: #0f172a; font-weight: 500; line-height: 1.6;">
                {html.escape(answer)}
            </div>
            <div style="font-size: 0.82rem; color: #475569; margin-top: 0.5rem; font-weight: 500;">
                The system verified available vector and knowledge graph indices and found insufficient grounded evidence.
            </div>
            <div style="font-size: 0.78rem; color: #92400e; margin-top: 0.6rem; border-top: 1px dashed #d97706; padding-top: 0.4rem; font-weight: 600;">
                💡 <b>RBAC Check:</b> If querying a restricted document, check your active <b>Department</b> and <b>Clearance Level</b> in the sidebar Control Panel to ensure you have permission to view it.
            </div>
        </div>
        """).strip()
        st.markdown(refusal_html, unsafe_allow_html=True)
    else:
        # Format [SOURCE X] citations nicely
        escaped_answer = html.escape(answer)
        formatted_answer = _format_source_tags(escaped_answer)

        ans_html = textwrap.dedent(f"""
        <div class="answer-box">
            <div class="answer-header">
                <span class="answer-header-title">Grounded Synthesis</span>
                <span class="model-badge">Engine: {html.escape(model_name or "LLM")}</span>
            </div>
            <div style="color: #0f172a; white-space: pre-wrap; font-size: 1.05rem; line-height: 1.65;">{formatted_answer}</div>
        </div>
        """).strip()
        st.markdown(ans_html, unsafe_allow_html=True)

    # Optional generation metrics pill
    gen_time = diagnostics.get("generation_time_sec")
    p_tokens = diagnostics.get("prompt_tokens")
    r_tokens = diagnostics.get("response_tokens")
    if gen_time is not None or p_tokens is not None:
        metrics_parts = []
        if gen_time is not None:
            metrics_parts.append(f"Latency: {gen_time:.2f}s")
        if p_tokens is not None and r_tokens is not None:
            metrics_parts.append(f"Tokens: {p_tokens} in / {r_tokens} out")
        st.caption(" &bull; ".join(metrics_parts))
