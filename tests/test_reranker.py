"""
Unit and Integration Tests for Step 5: Cross-Encoder Reranking.
"""
from typing import List, Dict, Any
from unittest.mock import MagicMock
import pytest

from app.models import SearchResult
from app.reranker import (
    CrossEncoderReranker,
    RerankedRetrievalPipeline,
    rerank_results,
    retrieve_and_rerank,
)


class MockCrossEncoderModel:
    """Mock CrossEncoder that returns predetermined scores for testing."""
    def __init__(self, scores: List[float]):
        self.scores = scores
        self.last_pairs = None
        self.last_batch_size = None

    def predict(self, pairs: List[List[str]], batch_size: int = 32):
        self.last_pairs = pairs
        self.last_batch_size = batch_size
        return self.scores[:len(pairs)]


# -------------------------------------------------------------------------
# Unit Tests for CrossEncoderReranker
# -------------------------------------------------------------------------

def test_reranker_reorders_by_score():
    """Verify candidates are reordered strictly by descending cross-encoder score."""
    candidates = [
        SearchResult(
            chunk_id="chunk_a",
            document_id="doc_1",
            content="Content A about database indexing",
            score=0.033,  # previous RRF score
            rank=1,
            rrf_score=0.033,
            metadata={"section": "Database"},
        ),
        SearchResult(
            chunk_id="chunk_b",
            document_id="doc_1",
            content="Content B about text chunking and overlap",
            score=0.025,
            rank=2,
            rrf_score=0.025,
            metadata={"section": "Chunking"},
        ),
        SearchResult(
            chunk_id="chunk_c",
            document_id="doc_2",
            content="Content C about Paris history",
            score=0.016,
            rank=3,
            rrf_score=0.016,
            metadata={"section": "Geography"},
        ),
    ]

    # Mock scores: Chunk A gets 0.20, Chunk B gets 0.95 (highest), Chunk C gets -0.80
    mock_model = MockCrossEncoderModel(scores=[0.20, 0.95, -0.80])
    reranker = CrossEncoderReranker(model=mock_model)

    reranked = reranker.rerank(
        query_text="What is chunking and overlap?",
        candidates=candidates,
        top_k=3,
    )

    assert len(reranked) == 3
    # Chunk B should be promoted to #1
    assert reranked[0].chunk_id == "chunk_b"
    assert reranked[0].rank == 1
    assert reranked[0].reranker_score == 0.95
    assert reranked[0].score == 0.95

    # Chunk A should be #2
    assert reranked[1].chunk_id == "chunk_a"
    assert reranked[1].rank == 2
    assert reranked[1].reranker_score == 0.20

    # Chunk C should be #3
    assert reranked[2].chunk_id == "chunk_c"
    assert reranked[2].rank == 3
    assert reranked[2].reranker_score == -0.80


def test_reranker_preserves_metadata_and_provenance():
    """Verify metadata, chunk_id, document_id, and original rrf_score are preserved."""
    candidates = [
        SearchResult(
            chunk_id="c_001",
            document_id="doc_rag",
            content="Structure-aware chunking keeps headers with content.",
            score=0.030,
            rank=1,
            rrf_score=0.030,
            metadata={"page_number": 3, "section": "Chunking", "source": "pdf"},
            sources={"vector": {"rank": 1, "score": 0.85}, "bm25": {"rank": 4, "score": 3.2}},
        )
    ]

    mock_model = MockCrossEncoderModel(scores=[4.56789])
    reranker = CrossEncoderReranker(model=mock_model)

    reranked = reranker.rerank(query_text="chunking", candidates=candidates, top_k=1)
    res = reranked[0]

    assert res.chunk_id == "c_001"
    assert res.document_id == "doc_rag"
    assert res.metadata["page_number"] == 3
    assert res.metadata["section"] == "Chunking"
    assert res.rrf_score == 0.030
    assert res.reranker_score == 4.56789
    assert res.sources["vector"]["rank"] == 1
    assert res.sources["bm25"]["rank"] == 4
    assert res.sources["reranker"]["rank"] == 1
    assert res.sources["reranker"]["score"] == 4.56789
    assert res.source == "reranked"


def test_reranker_accepts_dictionary_candidates():
    """Verify reranker seamlessly processes dictionary candidates (e.g. from hybrid retrieval)."""
    dict_candidates = [
        {
            "chunk_id": "dict_chunk_1",
            "document_id": "doc_1",
            "content": "First candidate text",
            "score": 0.02,
            "rank": 1,
            "rrf_score": 0.02,
            "metadata": {"section": "Alpha"},
            "sources": {"vector": {"rank": 1}},
        },
        {
            "chunk_id": "dict_chunk_2",
            "document_id": "doc_2",
            "content": "Second candidate text",
            "score": 0.015,
            "rank": 2,
            "rrf_score": 0.015,
            "metadata": {"section": "Beta"},
            "sources": {"bm25": {"rank": 1}},
        },
    ]

    mock_model = MockCrossEncoderModel(scores=[-2.0, 5.0])
    reranker = CrossEncoderReranker(model=mock_model)

    reranked = reranker.rerank(query_text="search query", candidates=dict_candidates, top_k=2)

    assert len(reranked) == 2
    assert isinstance(reranked[0], SearchResult)
    # Second candidate scored 5.0, so it must be rank 1
    assert reranked[0].chunk_id == "dict_chunk_2"
    assert reranked[0].rank == 1
    assert reranked[0].reranker_score == 5.0
    assert reranked[0].metadata["section"] == "Beta"

    assert reranked[1].chunk_id == "dict_chunk_1"
    assert reranked[1].rank == 2
    assert reranked[1].reranker_score == -2.0


def test_reranker_top_k_limiting():
    """Verify top_k parameter slices results correctly."""
    candidates = [
        SearchResult(chunk_id=f"c_{i}", document_id="d1", content=f"Text {i}")
        for i in range(5)
    ]
    # Scores: c_0=1.0, c_1=5.0, c_2=3.0, c_3=2.0, c_4=4.0
    mock_model = MockCrossEncoderModel(scores=[1.0, 5.0, 3.0, 2.0, 4.0])
    reranker = CrossEncoderReranker(model=mock_model)

    reranked_top2 = reranker.rerank(query_text="query", candidates=candidates, top_k=2)
    assert len(reranked_top2) == 2
    assert [r.chunk_id for r in reranked_top2] == ["c_1", "c_4"]
    assert [r.rank for r in reranked_top2] == [1, 2]


def test_reranker_candidates_smaller_than_top_k():
    """Verify if candidate pool is smaller than top_k, all candidates are returned reranked."""
    candidates = [
        SearchResult(chunk_id="c_1", document_id="d1", content="Text 1"),
        SearchResult(chunk_id="c_2", document_id="d1", content="Text 2"),
    ]
    mock_model = MockCrossEncoderModel(scores=[1.5, 3.5])
    reranker = CrossEncoderReranker(model=mock_model)

    reranked = reranker.rerank(query_text="query", candidates=candidates, top_k=10)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c_2"
    assert reranked[1].chunk_id == "c_1"


def test_reranker_edge_cases():
    """Verify empty list, top_k <= 0, and empty query edge cases."""
    mock_model = MockCrossEncoderModel(scores=[1.0])
    reranker = CrossEncoderReranker(model=mock_model)

    # Empty candidate list
    assert reranker.rerank(query_text="query", candidates=[], top_k=5) == []

    # top_k <= 0
    cands = [SearchResult(chunk_id="c_1", document_id="d1", content="Text")]
    assert reranker.rerank(query_text="query", candidates=cands, top_k=0) == []
    assert reranker.rerank(query_text="query", candidates=cands, top_k=-1) == []

    # Empty query raises ValueError
    with pytest.raises(ValueError, match="query_text cannot be empty"):
        reranker.rerank(query_text="", candidates=cands, top_k=5)
    with pytest.raises(ValueError, match="query_text cannot be empty"):
        reranker.rerank(query_text="   ", candidates=cands, top_k=5)


def test_reranker_passes_query_chunk_pairs_correctly():
    """Verify the exact pair structure [[query, chunk_text], ...] passed to model.predict."""
    candidates = [
        SearchResult(chunk_id="c_1", document_id="d1", content="Hello world"),
        SearchResult(chunk_id="c_2", document_id="d1", content="Second document"),
    ]
    mock_model = MockCrossEncoderModel(scores=[1.0, 2.0])
    reranker = CrossEncoderReranker(model=mock_model, batch_size=16)

    query = "test search"
    reranker.rerank(query_text=query, candidates=candidates, top_k=2)

    assert mock_model.last_pairs == [
        ["test search", "Hello world"],
        ["test search", "Second document"],
    ]
    assert mock_model.last_batch_size == 16


# -------------------------------------------------------------------------
# Unit Tests for RerankedRetrievalPipeline
# -------------------------------------------------------------------------

def test_pipeline_coordinates_stages():
    """Verify RerankedRetrievalPipeline calls hybrid retriever and feeds candidates to reranker."""
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        {"chunk_id": "c_1", "document_id": "d1", "content": "Chunk 1", "score": 0.03},
        {"chunk_id": "c_2", "document_id": "d1", "content": "Chunk 2", "score": 0.02},
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        SearchResult(chunk_id="c_2", document_id="d1", content="Chunk 2", rank=1, reranker_score=2.5),
        SearchResult(chunk_id="c_1", document_id="d1", content="Chunk 1", rank=2, reranker_score=0.5),
    ]

    pipeline = RerankedRetrievalPipeline(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        candidate_k=20,
        top_k=5,
    )

    results = pipeline.retrieve_and_rerank(query_text="what is chunking?", candidate_k=10, top_k=2)

    # Verify Stage 1 was invoked with candidate_k=10
    mock_hybrid.retrieve.assert_called_once_with(
        query_text="what is chunking?",
        top_k=10,
        access_context=None,
        conn=None,
    )

    # Verify Stage 2 was invoked with candidates and top_k=2
    mock_reranker.rerank.assert_called_once()
    args, kwargs = mock_reranker.rerank.call_args
    assert kwargs["query_text"] == "what is chunking?"
    assert len(kwargs["candidates"]) == 2
    assert kwargs["top_k"] == 2

    assert len(results) == 2
    assert results[0].chunk_id == "c_2"


def test_pipeline_retrieve_with_diagnostics():
    """Verify retrieve_with_diagnostics returns full diagnostic dictionary."""
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve_with_details.return_value = {
        "query": "test",
        "vector_results": [{"chunk_id": "c_1"}],
        "bm25_results": [{"chunk_id": "c_2"}],
        "fused_results": [{"chunk_id": "c_1"}, {"chunk_id": "c_2"}],
    }

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        SearchResult(chunk_id="c_1", document_id="d1", content="Text", rank=1, reranker_score=3.0)
    ]

    pipeline = RerankedRetrievalPipeline(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
    )

    diag = pipeline.retrieve_with_diagnostics(query_text="test")
    assert "vector_results" in diag
    assert "bm25_results" in diag
    assert "candidates" in diag
    assert "reranked_results" in diag
    assert diag["reranked_results"][0].chunk_id == "c_1"


def test_functional_convenience_helpers():
    """Verify rerank_results and retrieve_and_rerank helper functions work with custom instances."""
    mock_model = MockCrossEncoderModel(scores=[3.0, 1.0])
    reranker = CrossEncoderReranker(model=mock_model)

    candidates = [
        SearchResult(chunk_id="a", document_id="d1", content="A"),
        SearchResult(chunk_id="b", document_id="d1", content="B"),
    ]

    res = rerank_results("query", candidates, top_k=2, reranker=reranker)
    assert len(res) == 2
    assert res[0].chunk_id == "a"


# -------------------------------------------------------------------------
# Integration Test with Live Pretrained Model
# -------------------------------------------------------------------------

@pytest.mark.integration
def test_live_cross_encoder_scoring():
    """
    Live integration test: Verifies ms-marco-MiniLM-L-6-v2 ranks a semantically
    relevant passage significantly higher than an irrelevant passage.
    """
    query = "What is chunking and chunk overlap in RAG?"

    relevant_cand = SearchResult(
        chunk_id="c_relevant",
        document_id="doc_rag",
        content="Chunking splits large documents into smaller text segments. Chunk overlap preserves semantic context between boundaries.",
        metadata={"section": "Chunking"},
    )
    irrelevant_cand = SearchResult(
        chunk_id="c_irrelevant",
        document_id="doc_paris",
        content="The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
        metadata={"section": "Monuments"},
    )

    reranker = CrossEncoderReranker()
    results = reranker.rerank(query_text=query, candidates=[irrelevant_cand, relevant_cand], top_k=2)

    assert len(results) == 2
    # The relevant chunk must be ranked #1
    assert results[0].chunk_id == "c_relevant"
    assert results[1].chunk_id == "c_irrelevant"

    # Cross-encoder score difference must be substantial (e.g. > 5 points difference in logits)
    assert results[0].reranker_score > results[1].reranker_score
    assert (results[0].reranker_score - results[1].reranker_score) > 5.0
