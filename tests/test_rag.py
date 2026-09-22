"""
Unit Tests for End-to-End Enterprise RAG Pipeline (Step 6).
"""
import os
from unittest.mock import MagicMock
import pytest

from app.models import SearchResult, RAGResponse, Citation
from app.rag import RAGPipeline, answer_query
from app.prompts import NO_EVIDENCE_MESSAGE
from app.db import test_connection as check_db_connection


def test_rag_pipeline_end_to_end_mocked():
    """Verify complete coordination from query through retrieval, context, prompt, to answer."""
    # 1. Mock retrieval pipeline
    mock_retrieval = MagicMock()
    mock_chunks = [
        SearchResult(
            chunk_id=10,
            document_id="doc_arch",
            content="Chunk overlap preserves context across boundaries.",
            score=5.5,
            rank=1,
            reranker_score=5.5,
            rrf_score=0.032,
            metadata={"document_name": "architecture.pdf", "page_number": 2, "section": "Chunking"},
        ),
        SearchResult(
            chunk_id=13,
            document_id="doc_faq",
            content="Support FAQ recommends 800 characters per chunk.",
            score=3.2,
            rank=2,
            reranker_score=3.2,
            rrf_score=0.031,
            metadata={"document_name": "faq.txt", "page_number": 1, "section": "FAQ"},
        ),
    ]
    mock_retrieval.retrieve_with_diagnostics.return_value = {
        "candidates": mock_chunks,
        "reranked_results": mock_chunks,
    }

    # 2. Mock LLM provider
    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.generate.return_value = (
        "Chunk overlap preserves context across boundaries [SOURCE 1], "
        "and 800 characters is typical [SOURCE 2]."
    )

    pipeline = RAGPipeline(
        retrieval_pipeline=mock_retrieval,
        llm_provider=mock_llm,
        candidate_k=10,
        top_k=2,
    )

    response = pipeline.answer_query(query="What is chunk overlap?")

    assert isinstance(response, RAGResponse)
    assert response.query == "What is chunk overlap?"
    assert "Chunk overlap preserves context" in response.answer
    assert response.model_name == "gemini-2.5-flash"

    # Verify authoritative citations
    assert len(response.citations) == 2
    assert response.citations[0].source_id == 1
    assert response.citations[0].filename == "architecture.pdf"
    assert response.citations[0].page_number == 2
    assert response.citations[1].source_id == 2
    assert response.citations[1].filename == "faq.txt"

    # Verify LLM was called with grounded prompt containing [SOURCE 1] and [SOURCE 2]
    mock_llm.generate.assert_called_once()
    called_prompt = mock_llm.generate.call_args[1]["prompt"]
    assert "[SOURCE 1]" in called_prompt
    assert "[SOURCE 2]" in called_prompt
    assert "What is chunk overlap?" in called_prompt


def test_rag_pipeline_no_evidence_behavior():
    """Verify that when retrieval returns no candidates, pipeline returns standard refusal without calling LLM."""
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve_with_diagnostics.return_value = {
        "candidates": [],
        "reranked_results": [],
    }

    mock_llm = MagicMock()

    pipeline = RAGPipeline(
        retrieval_pipeline=mock_retrieval,
        llm_provider=mock_llm,
    )

    response = pipeline.answer_query(query="What is the capital of Mars?")

    # LLM must NOT be called to avoid hallucinations or waste
    mock_llm.generate.assert_not_called()

    assert response.answer == NO_EVIDENCE_MESSAGE
    assert response.citations == []
    assert response.retrieved_results == []


def test_rag_pipeline_empty_query_raises_error():
    """Verify empty query string raises ValueError."""
    pipeline = RAGPipeline(
        retrieval_pipeline=MagicMock(),
        llm_provider=MagicMock(),
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        pipeline.answer_query(query="")
    with pytest.raises(ValueError, match="Query cannot be empty"):
        pipeline.answer_query(query="   ")


def test_answer_query_functional_helper():
    """Verify answer_query convenience wrapper works with custom pipeline."""
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve_with_diagnostics.return_value = {
        "candidates": [],
        "reranked_results": [],
    }
    pipeline = RAGPipeline(retrieval_pipeline=mock_retrieval, llm_provider=MagicMock())

    response = answer_query("query", pipeline=pipeline)
    assert response.answer == NO_EVIDENCE_MESSAGE


@pytest.mark.skipif(
    (not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY")) or not check_db_connection(),
    reason="Requires live GEMINI_API_KEY and active PostgreSQL connection",
)
def test_live_rag_end_to_end():
    """Live end-to-end integration test if GEMINI_API_KEY is present and DB is online."""
    response = answer_query("What is chunking?")
    assert len(response.answer) > 0
    assert len(response.citations) > 0
