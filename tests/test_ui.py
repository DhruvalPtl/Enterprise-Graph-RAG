"""
Tests for UI API Client and Components (Phase G7).
"""
import pytest
from unittest.mock import patch, MagicMock
from ui.api_client import check_health, submit_query


def test_api_client_check_health_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "app": "Enterprise Knowledge Intelligence Platform API",
        "version": "1.0.0",
        "database": "connected",
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        res = check_health("http://localhost:8000")
        assert res["ok"] is True
        assert res["database"] == "connected"
        assert res["status"] == "ok"


def test_api_client_check_health_offline():
    import httpx

    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        res = check_health("http://localhost:8000")
        assert res["ok"] is False
        assert res["status"] == "offline"
        assert "Cannot connect" in res["error"]


def test_api_client_submit_query_validation():
    # Empty query rejection at client level
    res = submit_query("http://localhost:8000", query="   ")
    assert res["ok"] is False
    assert res["error_type"] == "client_validation"


def test_api_client_submit_query_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "query": "What is MMLU?",
        "answer": "MMLU is a benchmark [SOURCE 1].",
        "citations": [
            {
                "source_id": 1,
                "chunk_id": 100,
                "document_id": "doc1",
                "filename": "report.pdf",
                "page_number": 5,
                "retrieved_by": ["vector", "graph"],
                "content": "MMLU benchmark evaluated Gemini Ultra.",
            }
        ],
        "model_name": "gemini-3.5-flash-lite",
        "diagnostics": {
            "retrieval_mode": "hybrid_graph_vector",
            "vector_candidates_count": 10,
            "graph_candidates_count": 5,
            "combined_unique_count": 12,
            "reranked_count": 1,
            "graph_seed_entities": ["MMLU"],
            "graph_relationships": [
                {
                    "source": "Gemini Ultra",
                    "type": "EVALUATED_ON",
                    "target": "MMLU",
                    "document_id": "doc1",
                    "chunk_id": 100,
                    "page_number": 5,
                }
            ],
            "total_latency_ms": 320.5,
        },
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = submit_query("http://localhost:8000", query="What is MMLU?")
        assert res["ok"] is True
        data = res["data"]
        assert data["answer"] == "MMLU is a benchmark [SOURCE 1]."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["retrieved_by"] == ["vector", "graph"]
        assert len(data["diagnostics"]["graph_relationships"]) == 1
