"""
Unit Tests for FastAPI REST API.

Verifies:
- GET /health (liveness and database readiness)
- POST /query input validation (Pydantic models, string stripping, bounds, forbidden extras)
- Response schema validation against Citation and QueryResponse
- Shared RAGPipeline dependency injection and mocking
- Safe error handling (400, 422, 500, 502) without leaking secrets
- Interactive OpenAPI documentation (/docs, /openapi.json)
"""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.dependencies import set_rag_pipeline, reset_rag_pipeline
from app.models import RAGResponse, Citation, AccessContext
from app.prompts import NO_EVIDENCE_MESSAGE


@pytest.fixture(autouse=True)
def clean_pipeline_dependencies():
    """Ensure clean dependency state before and after each test."""
    reset_rag_pipeline()
    yield
    reset_rag_pipeline()


@pytest.fixture
def client():
    """Provides FastAPI TestClient instance."""
    return TestClient(app)


# -------------------------------------------------------------------------
# Health Endpoint Tests
# -------------------------------------------------------------------------

def test_health_endpoint_connected(client):
    """Verify /health returns 200 OK and database connected."""
    with patch("app.api.main.test_connection", return_value=True):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        assert "Enterprise Knowledge Intelligence Platform" in data["app"]
        assert data["version"] == "1.0.0"


def test_health_endpoint_degraded(client):
    """Verify /health returns degraded status when database is disconnected."""
    with patch("app.api.main.test_connection", return_value=False):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"] == "disconnected"


# -------------------------------------------------------------------------
# Query Endpoint Tests (POST /query)
# -------------------------------------------------------------------------

def test_query_endpoint_valid_request(client):
    """Verify valid POST /query request returns 200 with grounded answer and authoritative citations."""
    mock_pipeline = MagicMock()
    mock_response = RAGResponse(
        query="What is chunking?",
        answer="Chunking decomposes lengthy documents into focused passages [SOURCE 1].",
        citations=[
            Citation(
                source_id=1,
                chunk_id=10,
                document_id="doc_arch",
                filename="architecture.pdf",
                page_number=2,
                section="Chunking Strategy",
                reranker_score=5.5559,
                rrf_score=0.0325,
            )
        ],
        model_name="gemini-flash-latest",
        diagnostics={"candidate_count": 10, "evidence_count": 1},
    )
    mock_pipeline.answer_query.return_value = mock_response
    set_rag_pipeline(mock_pipeline)

    payload = {
        "query": "  What is chunking?  ",  # Whitespace will be stripped
        "top_k": 3,
    }
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "What is chunking?"
    assert "Chunking decomposes lengthy documents" in data["answer"]
    assert data["model_name"] == "gemini-flash-latest"

    # Verify citation structure
    assert len(data["citations"]) == 1
    c = data["citations"][0]
    assert c["source_id"] == 1
    assert c["filename"] == "architecture.pdf"
    assert c["page_number"] == 2
    assert c["section"] == "Chunking Strategy"
    assert c["reranker_score"] == 5.5559
    assert "[1] architecture.pdf" in c["formatted"]

    # Verify pipeline was called with normalized query and default public AccessContext
    mock_pipeline.answer_query.assert_called_once_with(
        query="What is chunking?",
        candidate_k=None,
        top_k=3,
        access_context=AccessContext(department="public", access_level="public", include_archived=False),
    )


def test_query_endpoint_empty_query_rejected(client):
    """Verify empty and whitespace-only queries return HTTP 422."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
    assert "validation failed" in response.json()["detail"].lower()

    response_ws = client.post("/query", json={"query": "    "})
    assert response_ws.status_code == 422


def test_query_endpoint_missing_query_field(client):
    """Verify missing required 'query' field returns HTTP 422."""
    response = client.post("/query", json={"top_k": 5})
    assert response.status_code == 422


def test_query_endpoint_invalid_top_k(client):
    """Verify invalid top_k bounds return HTTP 422."""
    # Zero
    assert client.post("/query", json={"query": "test", "top_k": 0}).status_code == 422
    # Negative
    assert client.post("/query", json={"query": "test", "top_k": -5}).status_code == 422
    # Exceeds max 50
    assert client.post("/query", json={"query": "test", "top_k": 100}).status_code == 422
    # String
    assert client.post("/query", json={"query": "test", "top_k": "invalid"}).status_code == 422


def test_query_endpoint_extra_forbidden_fields(client):
    """Verify unknown fields are rejected by extra='forbid'."""
    payload = {
        "query": "What is chunking?",
        "unsupported_extra_param": 123,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_endpoint_no_evidence_handling(client):
    """Verify query with no evidence returns 200 with standard refusal answer and empty citations."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.return_value = RAGResponse(
        query="What is the capital of Mars?",
        answer=NO_EVIDENCE_MESSAGE,
        citations=[],
        retrieved_results=[],
    )
    set_rag_pipeline(mock_pipeline)

    response = client.post("/query", json={"query": "What is the capital of Mars?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == NO_EVIDENCE_MESSAGE
    assert data["citations"] == []


def test_query_endpoint_internal_error_handling_sanitized(client):
    """Verify server exceptions return HTTP 500 without leaking stack traces or sensitive credentials."""
    mock_pipeline = MagicMock()
    fake_key = "AIza" + "Sy" + "TESTCREDENTIAL1234567890"
    mock_pipeline.answer_query.side_effect = RuntimeError(
        f"Database pool crash with secret key {fake_key} at postgres://admin:secret@host"
    )
    set_rag_pipeline(mock_pipeline)

    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 500
    data = response.json()

    # Verify no credentials leaked to client
    assert fake_key not in str(data)
    assert "secret@" not in str(data)
    assert "An internal server error occurred" in data["detail"]



def test_query_endpoint_upstream_llm_failure(client):
    """Verify upstream Gemini unavailable returns HTTP 502 Bad Gateway."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.side_effect = RuntimeError(
        "Gemini generation failed: 503 UNAVAILABLE. Model experiencing high demand."
    )
    set_rag_pipeline(mock_pipeline)

    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 502
    assert "Upstream LLM provider is temporarily unavailable" in response.json()["detail"]


# -------------------------------------------------------------------------
# Documentation Endpoint Tests (/docs, /openapi.json)
# -------------------------------------------------------------------------

def test_openapi_documentation(client):
    """Verify interactive Swagger UI and OpenAPI JSON endpoints are accessible."""
    docs_res = client.get("/docs")
    assert docs_res.status_code == 200
    assert "swagger" in docs_res.text.lower() or "html" in docs_res.text.lower()

    openapi_res = client.get("/openapi.json")
    assert openapi_res.status_code == 200
    schema = openapi_res.json()
    assert "/health" in schema["paths"]
    assert "/query" in schema["paths"]
    assert "QueryRequest" in schema["components"]["schemas"]
    assert "QueryResponse" in schema["components"]["schemas"]
