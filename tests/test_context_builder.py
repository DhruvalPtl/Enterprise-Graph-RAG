"""
Unit Tests for Context Builder & Authoritative Citations (Step 6).
"""
import pytest
from app.models import SearchResult, Citation
from app.context_builder import ContextBuilder, BuiltContext, build_context


def test_context_builder_formatting():
    """Verify structured formatting of retrieved chunks into [SOURCE X] blocks."""
    results = [
        SearchResult(
            chunk_id="chunk_10",
            document_id="doc_arch",
            content="Chunking splits large documents into cohesive paragraphs.",
            score=5.55,
            rank=1,
            reranker_score=5.55,
            rrf_score=0.0325,
            metadata={
                "document_name": "architecture.pdf",
                "page_number": 2,
                "section": "Architecture",
            },
            sources={"vector": {"rank": 2}, "bm25": {"rank": 1}},
        ),
        SearchResult(
            chunk_id="chunk_13",
            document_id="doc_faq",
            content="Support FAQ explains overlap and chunk size tuning.",
            score=-0.18,
            rank=2,
            reranker_score=-0.18,
            rrf_score=0.0325,
            metadata={
                "document_name": "faq.txt",
                "page_number": 1,
                "section": "FAQ",
            },
            sources={"vector": {"rank": 1}, "bm25": {"rank": 2}},
        ),
    ]

    builder = ContextBuilder(max_context_chars=4000)
    built = builder.build_context(query="What is chunking?", results=results)

    assert isinstance(built, BuiltContext)
    assert len(built.used_results) == 2
    assert len(built.citations) == 2

    # Check text formatting
    assert "[SOURCE 1]" in built.context_text
    assert "Document: architecture.pdf" in built.context_text
    assert "Page: 2" in built.context_text
    assert "Section: Architecture" in built.context_text
    assert "Chunk ID: chunk_10" in built.context_text
    assert "Chunking splits large documents into cohesive paragraphs." in built.context_text

    assert "[SOURCE 2]" in built.context_text
    assert "Document: faq.txt" in built.context_text
    assert "Page: 1" in built.context_text
    assert "Section: FAQ" in built.context_text
    assert "Chunk ID: chunk_13" in built.context_text


def test_citation_metadata_mapping():
    """Verify Citation objects match SearchResult metadata faithfully."""
    results = [
        SearchResult(
            chunk_id=42,
            document_id="doc_gov",
            content="PII must be redacted prior to storage.",
            score=2.34,
            rank=1,
            reranker_score=2.34,
            rrf_score=0.029,
            metadata={
                "document_name": "governance.md",
                "page_number": 3,
                "section": "Data Privacy",
            },
            sources={"vector": {"rank": 3}},
        )
    ]

    builder = ContextBuilder()
    built = builder.build_context(query="privacy", results=results)

    assert len(built.citations) == 1
    c = built.citations[0]
    assert c.source_id == 1
    assert c.chunk_id == 42
    assert c.document_id == "doc_gov"
    assert c.filename == "governance.md"
    assert c.page_number == 3
    assert c.section == "Data Privacy"
    assert c.reranker_score == 2.34
    assert c.rrf_score == 0.029

    formatted = c.format_citation()
    assert formatted == "[1] governance.md | Page 3 | Section: Data Privacy"


def test_context_budget_stops_adding_passages():
    """Verify ContextBuilder respects max_context_chars and keeps whole passages."""
    chunk1 = SearchResult(
        chunk_id="c1",
        document_id="d1",
        content="A" * 150,
        metadata={"document_name": "doc1.txt"},
    )
    chunk2 = SearchResult(
        chunk_id="c2",
        document_id="d2",
        content="B" * 150,
        metadata={"document_name": "doc2.txt"},
    )
    chunk3 = SearchResult(
        chunk_id="c3",
        document_id="d3",
        content="C" * 150,
        metadata={"document_name": "doc3.txt"},
    )

    # Each block is ~230 chars. Limit 350 allows chunk1, but not chunk1 + chunk2
    builder = ContextBuilder(max_context_chars=350)
    built = builder.build_context(query="test", results=[chunk1, chunk2, chunk3])

    assert len(built.used_results) == 1
    assert len(built.citations) == 1
    assert built.citations[0].chunk_id == "c1"
    assert "[SOURCE 1]" in built.context_text
    assert "[SOURCE 2]" not in built.context_text


def test_single_large_passage_truncation():
    """Verify that if even the first passage exceeds budget, it truncates gracefully."""
    huge_chunk = SearchResult(
        chunk_id="huge",
        document_id="doc_huge",
        content="Long content text that exceeds the small limit " * 20,
        metadata={"document_name": "huge.txt"},
    )

    builder = ContextBuilder(max_context_chars=120)
    built = builder.build_context(query="test", results=[huge_chunk])

    assert len(built.used_results) == 1
    assert len(built.citations) == 1
    assert built.total_chars <= 160
    assert "..." in built.context_text


def test_empty_results():
    """Verify empty results list yields empty context without errors."""
    builder = ContextBuilder()
    built = builder.build_context(query="anything", results=[])

    assert built.context_text == ""
    assert built.citations == []
    assert built.used_results == []
    assert built.total_chars == 0


def test_functional_build_context_helper():
    """Verify module-level build_context convenience helper."""
    results = [
        SearchResult(chunk_id="c1", document_id="d1", content="Sample content")
    ]
    built = build_context("query", results, max_context_chars=2000)
    assert len(built.citations) == 1
    assert built.citations[0].source_id == 1
