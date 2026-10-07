"""
Unit and Integration Tests for Phase G10.5-B Graph-Aware Reranking.

Tests:
1. Mode toggle ('baseline' vs 'graph_aware') via constructor, parameter, and env var.
2. Representation preservation: Vector-only candidates receive raw cand.content.
3. Representation augmentation: Graph-derived candidates receive compact [GRAPH PATH] header.
4. Baseline invariance: Graph candidates receive raw cand.content when mode='baseline'.
5. Non-mutation invariant: cand.content and candidate objects are not mutated in-place.
6. Compactness and deduplication of graph relationships.
7. Token overhead bounds validation (< 50 tokens overhead).
8. RBAC defense-in-depth: unauthorized graph candidates never reach reranker.
9. Hybrid retriever integration: reranking_mode propagation and metadata logging.
"""
import os
from unittest.mock import MagicMock, patch
import pytest

from app.models import SearchResult, AccessContext
from app.reranker import (
    CrossEncoderReranker,
    format_graph_context,
    build_reranker_candidate_text,
)
from app.hybrid_retriever import GraphVectorHybridRetriever


def test_format_graph_context():
    """Verify format_graph_context extracts clean, deduplicated relationship paths."""
    cand = SearchResult(
        chunk_id=101,
        document_id="DOC-01",
        content="Platform Engineering oversees Atlas services.",
        metadata={
            "graph_relationships": [
                {"source": "Atlas", "relationship_type": "DEVELOPED_BY", "target": "Platform Engineering"},
                {"source": "Atlas", "relationship_type": "DEVELOPED_BY", "target": "Platform Engineering"}, # Duplicate
                {"source": "Atlas", "relationship_type": "DEPLOYED_ON", "target": "AWS"},
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    context = format_graph_context(cand)
    assert context.startswith("[GRAPH PATH]")
    assert "Atlas --[DEVELOPED_BY]--> Platform Engineering" in context
    assert "Atlas --[DEPLOYED_ON]--> AWS" in context
    # Check deduplication
    assert context.count("DEVELOPED_BY") == 1


def test_build_reranker_candidate_text_baseline():
    """In baseline mode, candidates receive raw content regardless of graph provenance."""
    cand = SearchResult(
        chunk_id=101,
        document_id="DOC-01",
        content="Platform Engineering oversees Atlas services.",
        metadata={
            "graph_relationships": [
                {"source": "Atlas", "relationship_type": "DEVELOPED_BY", "target": "Platform Engineering"},
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    text = build_reranker_candidate_text(cand, reranking_mode="baseline")
    assert text == cand.content
    assert "[GRAPH PATH]" not in text


def test_build_reranker_candidate_text_vector_in_graph_aware():
    """In graph_aware mode, vector-only candidates MUST remain untouched."""
    cand = SearchResult(
        chunk_id=102,
        document_id="DOC-02",
        content="This is standard documentation retrieved by vector search.",
        metadata={"department": "engineering"},
        score=0.85,
        rank=1,
        source="vector",
        sources={"retrieved_by": ["vector"]},
    )

    text = build_reranker_candidate_text(cand, reranking_mode="graph_aware")
    assert text == cand.content
    assert "[GRAPH PATH]" not in text


def test_build_reranker_candidate_text_graph_in_graph_aware():
    """In graph_aware mode, graph candidates receive compact relationship header."""
    cand = SearchResult(
        chunk_id=101,
        document_id="DOC-01",
        content="Platform Engineering oversees Atlas services.",
        metadata={
            "graph_relationships": [
                {"source": "Atlas", "relationship_type": "DEVELOPED_BY", "target": "Platform Engineering"},
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    text = build_reranker_candidate_text(cand, reranking_mode="graph_aware")
    assert "[GRAPH PATH]" in text
    assert "Atlas --[DEVELOPED_BY]--> Platform Engineering" in text
    assert "[DOCUMENT CONTENT]" in text
    assert cand.content in text


def test_candidate_content_immutability():
    """Verify cand.content is never mutated in-place by the reranker."""
    original_content = "Raw chunk content that must remain untouched."
    cand = SearchResult(
        chunk_id=103,
        document_id="DOC-03",
        content=original_content,
        metadata={
            "graph_relationships": [
                {"source": "EntityA", "relationship_type": "RELATES_TO", "target": "EntityB"}
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = [2.5]
    reranker = CrossEncoderReranker(model=mock_model, reranking_mode="graph_aware")

    results = reranker.rerank(query_text="What is EntityA?", candidates=[cand], top_k=1)

    # Invariant: cand.content and results[0].content must be identical to original_content
    assert cand.content == original_content
    assert results[0].content == original_content
    assert results[0].sources["reranker"]["graph_augmented"] is True
    assert results[0].sources["reranker"]["reranking_mode"] == "graph_aware"


def test_reranker_mode_precedence():
    """Test runtime parameter overrides instance and env settings."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [1.0]
    reranker = CrossEncoderReranker(model=mock_model, reranking_mode="baseline")

    cand = SearchResult(
        chunk_id=104,
        document_id="DOC-04",
        content="Candidate text.",
        metadata={
            "graph_relationships": [
                {"source": "Src", "relationship_type": "EDGE", "target": "Dst"}
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    # 1. Baseline by default
    reranker.rerank(query_text="test query", candidates=[cand])
    call_args_baseline = mock_model.predict.call_args[0][0]
    assert call_args_baseline[0][1] == "Candidate text."

    # 2. Graph-aware override
    reranker.rerank(query_text="test query", candidates=[cand], reranking_mode="graph_aware")
    call_args_graph = mock_model.predict.call_args[0][0]
    assert "[GRAPH PATH]" in call_args_graph[0][1]
    assert "Src --[EDGE]--> Dst" in call_args_graph[0][1]


def test_token_overhead_compactness():
    """Verify that graph context adds compact token overhead (< 50 tokens)."""
    cand = SearchResult(
        chunk_id=105,
        document_id="DOC-05",
        content="Platform Engineering oversees Atlas services across clusters.",
        metadata={
            "graph_relationships": [
                {"source": "Atlas", "relationship_type": "DEVELOPED_BY", "target": "Platform Engineering"},
                {"source": "Atlas", "relationship_type": "USES", "target": "Payment Service"},
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    base_text = build_reranker_candidate_text(cand, "baseline")
    graph_text = build_reranker_candidate_text(cand, "graph_aware")

    # Character and word length ratio
    base_words = len(base_text.split())
    graph_words = len(graph_text.split())
    additional_words = graph_words - base_words

    # Compactness check: header introduces roughly 10-25 words
    assert additional_words < 40


def test_hybrid_retriever_mode_propagation():
    """Verify GraphVectorHybridRetriever propagates reranking_mode and logs telemetry."""
    mock_reranker = MagicMock()
    mock_reranked_pipeline = MagicMock()
    mock_reranked_pipeline.hybrid_retriever.retrieve_with_details.return_value = {
        "fused_results": [
            SearchResult(
                chunk_id=201,
                document_id="DOC-VEC",
                content="Vector content",
                score=0.9,
                rank=1,
                source="vector",
                sources={"retrieved_by": ["vector"]},
            )
        ]
    }
    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = None

    retriever = GraphVectorHybridRetriever(
        reranked_pipeline=mock_reranked_pipeline,
        graph_retriever=mock_graph_retriever,
        reranker=mock_reranker,
        reranking_mode="graph_aware",
    )

    mock_reranker.rerank.return_value = []
    res = retriever.retrieve_fused(query="test query")

    assert res.retrieval_metadata["reranking_mode"] == "graph_aware"
    mock_reranker.rerank.assert_called_once()
    _, kwargs = mock_reranker.rerank.call_args
    assert kwargs.get("reranking_mode") == "graph_aware"


def test_rbac_defense_in_depth_no_leakage_to_reranker():
    """
    Verify that candidates failing RBAC clearance are pruned BEFORE reaching
    the cross-encoder reranker, ensuring unauthorized graph context is never exposed.
    """
    # Create an unauthorized candidate (confidential manager clearance)
    unauthorized_cand = SearchResult(
        chunk_id=301,
        document_id="DOC-SECRET",
        content="Confidential acquisition details.",
        metadata={
            "department": "finance",
            "access_level": "manager",
            "status": "active",
            "graph_relationships": [
                {"source": "NovaTech", "relationship_type": "ACQUIRING", "target": "SecretVendor"}
            ]
        },
        score=1.0,
        rank=1,
        source="graph",
        sources={"retrieved_by": ["graph"]},
    )

    authorized_cand = SearchResult(
        chunk_id=302,
        document_id="DOC-PUB",
        content="Public financial summary.",
        metadata={
            "department": "finance",
            "access_level": "public",
            "status": "active",
        },
        score=0.8,
        rank=2,
        source="vector",
        sources={"retrieved_by": ["vector"]},
    )

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [authorized_cand]

    mock_pipeline = MagicMock()
    mock_pipeline.hybrid_retriever.retrieve_with_details.return_value = {
        "fused_results": [authorized_cand]
    }
    mock_graph_retriever = MagicMock()
    # Mock graph retriever returning the unauthorized chunk
    mock_graph_retriever.retrieve.return_value = MagicMock(
        relationships=[],
        matched_entities=[],
        connected_entities=[],
        source_chunks=[
            {
                "chunk_id": 301,
                "document_id": "DOC-SECRET",
                "content": "Confidential acquisition details.",
                "department": "finance",
                "access_level": "manager",
                "status": "active",
            }
        ]
    )

    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=mock_pipeline,
        graph_retriever=mock_graph_retriever,
        reranker=mock_reranker,
        reranking_mode="graph_aware",
    )

    # Caller has public finance access
    ctx = AccessContext(department="finance", access_level="public")
    res = hybrid.retrieve_fused("acquisition status", access_context=ctx)

    # Check candidates passed to reranker
    candidates_passed = mock_reranker.rerank.call_args[1]["candidates"]
    passed_ids = [c.chunk_id for c in candidates_passed]

    # Invariant: Chunk 301 MUST NOT reach the reranker
    assert 301 not in passed_ids
    assert 302 in passed_ids
