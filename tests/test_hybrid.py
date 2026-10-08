"""
Unit and Integration Tests for Hybrid Retrieval.
"""
import pytest
from unittest.mock import MagicMock

from app.hybrid import HybridRetriever, hybrid_retrieve
from app.db import test_connection as check_db_connection


# --- 1. Unit Tests with Mocks ---

def test_hybrid_empty_query_raises_error():
    """Verifies ValueError when query is empty or whitespace."""
    retriever = HybridRetriever(vector_store=MagicMock(), bm25_retriever=MagicMock())
    with pytest.raises(ValueError, match="query_text cannot be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="query_text cannot be empty"):
        retriever.retrieve("   \n\t  ")


def test_hybrid_delegates_to_both_retrievers():
    """Verifies both vector_store and bm25_retriever are invoked independently."""
    mock_vector = MagicMock()
    mock_bm25 = MagicMock()

    mock_vector.retrieve.return_value = [
        {"chunk_id": "c1", "rank": 1, "source": "vector", "score": 0.88, "content": "Chunk 1"}
    ]
    mock_bm25.search.return_value = [
        {"chunk_id": "c2", "rank": 1, "source": "bm25", "score": 3.45, "content": "Chunk 2"}
    ]

    retriever = HybridRetriever(
        vector_store=mock_vector,
        bm25_retriever=mock_bm25,
        vector_top_k=10,
        bm25_top_k=10,
        rrf_top_k=5,
    )

    results = retriever.retrieve("Test search query", top_k=5)

    # Verify both systems were queried with the query text
    mock_vector.retrieve.assert_called_once_with(query_text="Test search query", top_k=10, access_context=None, conn=None)
    mock_bm25.search.assert_called_once_with(query_text="Test search query", top_k=10, access_context=None)

    # Verify fusion produced results containing both candidates
    assert len(results) == 2
    ids = [r["chunk_id"] for r in results]
    assert "c1" in ids
    assert "c2" in ids
    assert all(r["source"] == "hybrid" for r in results)


def test_hybrid_retrieve_with_details():
    """Verifies retrieve_with_details exposes individual rankings and fused list."""
    mock_vector = MagicMock()
    mock_bm25 = MagicMock()

    mock_vector.retrieve.return_value = [{"chunk_id": "v1", "rank": 1, "source": "vector"}]
    mock_bm25.search.return_value = [{"chunk_id": "b1", "rank": 1, "source": "bm25"}]

    retriever = HybridRetriever(vector_store=mock_vector, bm25_retriever=mock_bm25)
    details = retriever.retrieve_with_details("Audit compliance", top_k=3)

    assert "query" in details
    assert "vector_results" in details
    assert "bm25_results" in details
    assert "fused_results" in details
    assert details["query"] == "Audit compliance"
    assert len(details["vector_results"]) == 1
    assert len(details["bm25_results"]) == 1
    assert len(details["fused_results"]) == 2


def test_hybrid_reinforcement_for_common_chunk():
    """
    Verifies that a chunk appearing in both vector and BM25 rankings
    ranks higher than a chunk appearing in only one system.
    """
    mock_vector = MagicMock()
    mock_bm25 = MagicMock()

    # Chunk 'SHARED' is #1 in Vector and #1 in BM25
    # Chunk 'V_ONLY' is #2 in Vector
    mock_vector.retrieve.return_value = [
        {"chunk_id": "SHARED", "rank": 1, "source": "vector"},
        {"chunk_id": "V_ONLY", "rank": 2, "source": "vector"},
    ]
    # Chunk 'B_ONLY' is #2 in BM25
    mock_bm25.search.return_value = [
        {"chunk_id": "SHARED", "rank": 1, "source": "bm25"},
        {"chunk_id": "B_ONLY", "rank": 2, "source": "bm25"},
    ]

    retriever = HybridRetriever(vector_store=mock_vector, bm25_retriever=mock_bm25, rrf_k=60)
    results = retriever.retrieve("Testing reinforcement", top_k=3)

    assert len(results) == 3
    # SHARED must rank #1 with score ~ 2/(60+1)
    assert results[0]["chunk_id"] == "SHARED"
    assert results[0]["rrf_score"] == pytest.approx(2.0 / 61.0, abs=1e-5)
    assert "vector" in results[0]["sources"]
    assert "bm25" in results[0]["sources"]


def test_hybrid_retrieve_functional_helper():
    """Verifies the standalone hybrid_retrieve() helper function."""
    mock_vector = MagicMock()
    mock_bm25 = MagicMock()
    mock_vector.retrieve.return_value = [{"chunk_id": 1, "rank": 1, "source": "vector"}]
    mock_bm25.search.return_value = [{"chunk_id": 2, "rank": 1, "source": "bm25"}]

    custom_retriever = HybridRetriever(vector_store=mock_vector, bm25_retriever=mock_bm25)
    results = hybrid_retrieve("test query", top_k=2, hybrid_retriever=custom_retriever)
    assert len(results) == 2


# --- 2. Live Integration Tests against PostgreSQL and Live Corpus ---

IS_POSTGRES_AVAILABLE = check_db_connection()


@pytest.mark.skipif(
    not IS_POSTGRES_AVAILABLE,
    reason="Live PostgreSQL instance not detected on localhost:5432. Live hybrid integration skipped.",
)
def test_live_hybrid_retrieval_against_database():
    """Verifies live end-to-end hybrid retrieval against PostgreSQL and BM25 index."""
    retriever = HybridRetriever(
        vector_top_k=5,
        bm25_top_k=5,
        rrf_top_k=3,
    )

    query = "What is chunking and chunk overlap strategy?"
    results = retriever.retrieve(query_text=query, top_k=3)

    assert len(results) >= 1
    top = results[0]
    assert "chunk_id" in top
    assert "document_id" in top
    assert "content" in top
    assert "rrf_score" in top
    assert top["rrf_score"] > 0.0
    assert top["rank"] == 1
    assert "sources" in top
    assert top["metadata"]["document_name"] is not None
