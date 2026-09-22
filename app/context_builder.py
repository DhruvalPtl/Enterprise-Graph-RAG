"""
Context Builder for Grounded Enterprise RAG Generation (Step 6).

This module prepares retrieved and reranked passages into a structured, bounded
context block for the LLM prompt, and maps each passage to an authoritative Citation.

Responsibilities:
1. Formats chunks with explicit, numbered [SOURCE X] identifiers.
2. Extracts authoritative document metadata (filename, page, section, chunk_id).
3. Preserves descending cross-encoder relevance ordering.
4. Enforces character context budgets (MAX_CONTEXT_CHARS) while keeping passages intact.
5. Does NOT call the LLM, modify source text, or invent metadata.
"""
from typing import List, Tuple, Optional, Any, Dict
from dataclasses import dataclass, field

from app.config import MAX_CONTEXT_CHARS
from app.models import SearchResult, Citation


@dataclass
class BuiltContext:
    """Container for constructed context text and corresponding authoritative citations."""
    context_text: str
    citations: List[Citation]
    used_results: List[SearchResult]
    total_chars: int


class ContextBuilder:
    """
    Constructs a structured, bounded context string for the LLM prompt
    from a list of reranked SearchResult candidates.
    """

    def __init__(self, max_context_chars: int = MAX_CONTEXT_CHARS):
        """
        Initialize ContextBuilder.

        Args:
            max_context_chars: Upper bound for context length in characters.
        """
        self.max_context_chars = max_context_chars

    def build_context(
        self,
        query: str,
        results: List[SearchResult],
        max_context_chars: Optional[int] = None,
    ) -> BuiltContext:
        """
        Formats retrieved SearchResult candidates into a structured context string
        and builds authoritative Citation records.

        Args:
            query: User's original question or search query.
            results: List of SearchResult objects (ordered by relevance).
            max_context_chars: Optional override for maximum context length.

        Returns:
            BuiltContext containing the structured text and citations.
        """
        limit = max_context_chars if max_context_chars is not None else self.max_context_chars

        if not results:
            return BuiltContext(
                context_text="",
                citations=[],
                used_results=[],
                total_chars=0,
            )

        context_blocks: List[str] = []
        citations: List[Citation] = []
        used_results: List[SearchResult] = []
        current_length = 0

        for idx, res in enumerate(results, start=1):
            metadata = res.metadata or {}
            filename = (
                metadata.get("document_name")
                or metadata.get("filename")
                or str(res.document_id)
            )
            page_number = metadata.get("page_number")
            section = metadata.get("section", "General")
            content = (res.content or "").strip()

            # Header details
            header_lines = [
                f"[SOURCE {idx}]",
                f"Document: {filename}",
            ]
            if page_number is not None:
                header_lines.append(f"Page: {page_number}")
            if section and section != "General":
                header_lines.append(f"Section: {section}")
            header_lines.append(f"Chunk ID: {res.chunk_id}")

            # Compact graph relationship provenance (Phase G4)
            graph_rels = metadata.get("graph_relationships") or res.sources.get("graph", {}).get("relationships")
            if graph_rels:
                for gr in graph_rels[:3]:
                    s_name = gr.get("source")
                    r_type = gr.get("relationship_type")
                    t_name = gr.get("target")
                    if s_name and r_type and t_name:
                        header_lines.append(f"[GRAPH RELATIONSHIP] {s_name} --[{r_type}]--> {t_name}")

            header_str = "\n".join(header_lines)
            block_str = f"{header_str}\n\nContent:\n{content}\n"

            # Check context budget
            block_len = len(block_str) + (2 if context_blocks else 0)  # separator "\n\n"
            if current_length + block_len > limit:
                # If even the very first block exceeds limit, truncate content to fit
                if not context_blocks:
                    overhead = len(header_str) + len("\n\nContent:\n\n")
                    avail = max(50, limit - overhead)
                    truncated_content = content[:avail] + "..."
                    block_str = f"{header_str}\n\nContent:\n{truncated_content}\n"
                else:
                    # Stop adding further passages to keep whole, coherent passages
                    break

            context_blocks.append(block_str)
            current_length += len(block_str) + (2 if len(context_blocks) > 1 else 0)
            used_results.append(res)

            # Create matching authoritative Citation
            citation = Citation(
                source_id=idx,
                chunk_id=res.chunk_id,
                document_id=res.document_id,
                filename=filename,
                page_number=page_number,
                section=section,
                reranker_score=res.reranker_score,
                rrf_score=res.rrf_score,
                sources={
                    **dict(res.sources),
                    "content": res.content,
                    "retrieved_by": res.sources.get("retrieved_by", ["vector"]),
                },
            )
            citations.append(citation)

        full_context_text = "\n\n".join(context_blocks)

        return BuiltContext(
            context_text=full_context_text,
            citations=citations,
            used_results=used_results,
            total_chars=len(full_context_text),
        )


def build_context(
    query: str,
    results: List[SearchResult],
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> BuiltContext:
    """Functional convenience helper to build structured context."""
    builder = ContextBuilder(max_context_chars=max_context_chars)
    return builder.build_context(query=query, results=results, max_context_chars=max_context_chars)
