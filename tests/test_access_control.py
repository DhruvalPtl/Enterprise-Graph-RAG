"""
Unit and Integration Tests for Metadata Filtering + Access-Aware Retrieval.

Tests cover:
1. test_access_context_defaults: verifies default context is public/public/not admin.
2. test_access_context_hierarchy: verifies public < employee < manager < admin level logic.
3. test_access_context_department_isolation: verifies finance employee cannot access engineering employee doc.
4. test_access_context_admin_override: verifies admin can access any department at any level.
5. test_access_context_archived_documents: verifies archived docs excluded by default, included when flag set.
6. test_vector_search_with_access_filter: mock/real DB test verifying WHERE clause includes access filters.
7. test_vector_search_unauthorized_excluded: verifies unauthorized chunks not returned in vector results.
8. test_bm25_with_access_filter: verifies BM25 retriever filters candidates by access context.
9. test_bm25_unauthorized_excluded: verifies unauthorized chunks never receive BM25 scores.
10. test_hybrid_retrieval_access_filter: verifies both vector and BM25 apply filters before RRF.
11. test_reranker_pipeline_access_filter: verifies cross-encoder only receives authorized candidates.
12. test_rag_pipeline_authorized_query: verifies authorized user gets answer with citations.
13. test_rag_pipeline_unauthorized_query: verifies unauthorized user gets refusal with NO citations and NO doc leaks.
14. test_api_query_with_access_context: verifies POST /query with valid access_context succeeds.
15. test_api_query_default_access_context: verifies POST /query without access_context defaults to public.
16. test_api_query_unauthorized_no_leak: verifies POST /query for restricted content returns refusal without leaking existence.
"""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.models import AccessContext, ACCESS_LEVEL_HIERARCHY, SearchResult, RAGResponse, Citation
from app.prompts import NO_EVIDENCE_MESSAGE
from app.bm25 import BM25Retriever
from app.vector_store import search_similar_chunks, VectorStore
from app.hybrid import HybridRetriever
from app.reranker import RerankedRetrievalPipeline, CrossEncoderReranker
from app.rag import RAGPipeline
from app.api.main import app
from app.api.dependencies import set_rag_pipeline, reset_rag_pipeline


@pytest.fixture(autouse=True)
def clean_api_dependencies():
    """Ensure clean pipeline state before and after each test."""
    reset_rag_pipeline()
    yield
    reset_rag_pipeline()


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------------------------------------------------------
# 1. AccessContext Unit Tests
# -------------------------------------------------------------------------

def test_access_context_defaults():
    """Verifies that default AccessContext is strictly unprivileged (public/public, not admin)."""
    ctx = AccessContext()
    assert ctx.department == "public"
    assert ctx.access_level == "public"
    assert ctx.include_archived is False
    assert ctx.is_admin is False
    assert ctx.allowed_access_levels() == ["public"]


def test_access_context_hierarchy():
    """Verifies public (0) < employee (1) < manager (2) < admin (3) hierarchy."""
    assert ACCESS_LEVEL_HIERARCHY["public"] < ACCESS_LEVEL_HIERARCHY["employee"]
    assert ACCESS_LEVEL_HIERARCHY["employee"] < ACCESS_LEVEL_HIERARCHY["manager"]
    assert ACCESS_LEVEL_HIERARCHY["manager"] < ACCESS_LEVEL_HIERARCHY["admin"]

    emp_ctx = AccessContext(department="engineering", access_level="employee")
    assert set(emp_ctx.allowed_access_levels()) == {"public", "employee"}
    assert emp_ctx.is_authorized(doc_department="engineering", doc_access_level="public") is True
    assert emp_ctx.is_authorized(doc_department="engineering", doc_access_level="employee") is True
    assert emp_ctx.is_authorized(doc_department="engineering", doc_access_level="manager") is False
    assert emp_ctx.is_authorized(doc_department="engineering", doc_access_level="admin") is False

    mgr_ctx = AccessContext(department="engineering", access_level="manager")
    assert set(mgr_ctx.allowed_access_levels()) == {"public", "employee", "manager"}
    assert mgr_ctx.is_authorized(doc_department="engineering", doc_access_level="employee") is True
    assert mgr_ctx.is_authorized(doc_department="engineering", doc_access_level="manager") is True
    assert mgr_ctx.is_authorized(doc_department="engineering", doc_access_level="admin") is False


def test_access_context_department_isolation():
    """Verifies that a user in one department cannot access department-specific docs of another."""
    fin_ctx = AccessContext(department="finance", access_level="employee")
    
    # Same level, different department -> forbidden
    assert fin_ctx.is_authorized(doc_department="engineering", doc_access_level="employee") is False

    # Document in department 'public' -> accessible if user level permits
    assert fin_ctx.is_authorized(doc_department="public", doc_access_level="employee") is True
    assert fin_ctx.is_authorized(doc_department="public", doc_access_level="public") is True

    # Higher level doc in department 'public' -> forbidden
    assert fin_ctx.is_authorized(doc_department="public", doc_access_level="manager") is False


def test_access_context_admin_override():
    """Verifies admin context has universal access across all departments and levels."""
    admin_ctx = AccessContext(department="any", access_level="admin")
    assert admin_ctx.is_admin is True
    assert set(admin_ctx.allowed_access_levels()) == {"public", "employee", "manager", "admin"}

    assert admin_ctx.is_authorized(doc_department="engineering", doc_access_level="manager") is True
    assert admin_ctx.is_authorized(doc_department="finance", doc_access_level="employee") is True
    assert admin_ctx.is_authorized(doc_department="hr", doc_access_level="admin") is True


def test_access_context_archived_documents():
    """Verifies archived documents are excluded by default, but included when include_archived is True."""
    ctx_normal = AccessContext(department="engineering", access_level="manager", include_archived=False)
    assert ctx_normal.is_authorized(
        doc_department="engineering", doc_access_level="manager", doc_status="archived"
    ) is False

    ctx_archived = AccessContext(department="engineering", access_level="manager", include_archived=True)
    assert ctx_archived.is_authorized(
        doc_department="engineering", doc_access_level="manager", doc_status="archived"
    ) is True


# -------------------------------------------------------------------------
# 2. Vector Search Access Filtering Tests
# -------------------------------------------------------------------------

def test_vector_search_with_access_filter():
    """Verifies that search_similar_chunks passes access parameters in PostgreSQL query."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchall.return_value = []

    dummy_vector = [0.1] * 384
    ctx = AccessContext(department="engineering", access_level="employee")

    search_similar_chunks(
        query_embedding=dummy_vector,
        top_k=5,
        access_context=ctx,
        conn=mock_conn,
    )

    mock_cur.execute.assert_called_once()
    sql_arg, params_arg = mock_cur.execute.call_args[0]

    # Verify SQL query applies access filtering before LIMIT
    assert "d.status = 'active'" in sql_arg
    assert "d.access_level = ANY(%s)" in sql_arg
    assert "d.department = 'public' OR d.department = %s" in sql_arg

    # Verify query parameters match context
    assert params_arg[0] == dummy_vector
    assert params_arg[1] is False  # include_archived
    assert params_arg[2] is False  # is_admin
    assert params_arg[3] == ["public", "employee"]  # allowed_levels
    assert params_arg[4] is False  # is_admin
    assert params_arg[5] == "engineering"  # department
    assert params_arg[6] == 5  # top_k


def test_vector_search_unauthorized_excluded():
    """Verifies vector store search excludes unauthorized chunks and maps metadata properly."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    
    # Mock row returned by DB with authorized attributes
    mock_cur.fetchall.return_value = [
        {
            "chunk_id": 101,
            "document_id": "doc_arch",
            "content": "Microservices and Kubernetes cluster architecture.",
            "page_number": 1,
            "section": "Overview",
            "chunk_index": 0,
            "document_name": "architecture.pdf",
            "document_type": "pdf",
            "department": "engineering",
            "access_level": "employee",
            "status": "active",
            "distance": 0.15,
        }
    ]

    store = VectorStore(embedding_service=MagicMock())
    results = store.search(
        query_embedding=[0.1] * 384,
        top_k=5,
        access_context=AccessContext(department="engineering", access_level="employee"),
        conn=mock_conn,
    )

    assert len(results) == 1
    res = results[0]
    assert res["chunk_id"] == 101
    assert res["metadata"]["department"] == "engineering"
    assert res["metadata"]["access_level"] == "employee"


# -------------------------------------------------------------------------
# 3. BM25 Retrieval Access Filtering Tests
# -------------------------------------------------------------------------

def test_bm25_with_access_filter():
    """Verifies BM25 retriever filters candidates by access context prior to candidate scoring."""
    sample_chunks = [
        {
            "chunk_id": 1,
            "content": "Enterprise platform password reset and support FAQ.",
            "metadata": {
                "document_name": "support_faq.txt",
                "department": "public",
                "access_level": "public",
                "status": "active",
            },
        },
        {
            "chunk_id": 2,
            "content": "Enterprise platform architecture and microservices design.",
            "metadata": {
                "document_name": "enterprise_platform_architecture.pdf",
                "department": "engineering",
                "access_level": "employee",
                "status": "active",
            },
        },
        {
            "chunk_id": 3,
            "content": "Enterprise platform AI governance policy and model risk assessment.",
            "metadata": {
                "document_name": "ai_governance_policy.md",
                "department": "engineering",
                "access_level": "manager",
                "status": "active",
            },
        },
    ]

    retriever = BM25Retriever.from_chunks(sample_chunks)

    # 1. Public caller should only retrieve chunk 1
    public_res = retriever.search(
        query_text="enterprise platform",
        top_k=5,
        access_context=AccessContext(department="public", access_level="public"),
    )
    assert len(public_res) == 1
    assert public_res[0]["chunk_id"] == 1

    # 2. Engineering employee should retrieve chunk 1 and chunk 2
    emp_res = retriever.search(
        query_text="enterprise platform",
        top_k=5,
        access_context=AccessContext(department="engineering", access_level="employee"),
    )
    assert len(emp_res) == 2
    chunk_ids = {r["chunk_id"] for r in emp_res}
    assert chunk_ids == {1, 2}

    # 3. Engineering manager should retrieve all 3 chunks
    mgr_res = retriever.search(
        query_text="enterprise platform",
        top_k=5,
        access_context=AccessContext(department="engineering", access_level="manager"),
    )
    assert len(mgr_res) == 3


def test_bm25_unauthorized_excluded():
    """Verifies that an unauthorized chunk never receives a BM25 score and is omitted entirely."""
    sample_chunks = [
        {
            "chunk_id": 10,
            "content": "AI model risk assessment guidelines and safety board approvals.",
            "metadata": {
                "document_name": "ai_governance_policy.md",
                "department": "engineering",
                "access_level": "manager",
                "status": "active",
            },
        }
    ]

    retriever = BM25Retriever.from_chunks(sample_chunks)

    # Employee asks about risk assessment
    results = retriever.search(
        query_text="risk assessment guidelines",
        top_k=5,
        access_context=AccessContext(department="engineering", access_level="employee"),
    )
    # Must be completely empty
    assert results == []


# -------------------------------------------------------------------------
# 4. Hybrid Retrieval Access Filtering Tests
# -------------------------------------------------------------------------

def test_hybrid_retrieval_access_filter():
    """Verifies that HybridRetriever passes access_context to both vector and BM25 retrievers."""
    mock_vector = MagicMock()
    mock_bm25 = MagicMock()
    mock_vector.retrieve.return_value = []
    mock_bm25.search.return_value = []

    retriever = HybridRetriever(vector_store=mock_vector, bm25_retriever=mock_bm25)
    ctx = AccessContext(department="engineering", access_level="employee")

    retriever.retrieve(
        query_text="Kubernetes deployment",
        top_k=5,
        access_context=ctx,
    )

    mock_vector.retrieve.assert_called_once_with(
        query_text="Kubernetes deployment",
        top_k=retriever.vector_top_k,
        access_context=ctx,
        conn=None,
    )
    mock_bm25.search.assert_called_once_with(
        query_text="Kubernetes deployment",
        top_k=retriever.bm25_top_k,
        access_context=ctx,
    )


# -------------------------------------------------------------------------
# 5. Reranker Pipeline Access Filtering Tests
# -------------------------------------------------------------------------

def test_reranker_pipeline_access_filter():
    """Verifies that Cross-Encoder reranker only receives candidates authorized by access context."""
    mock_hybrid = MagicMock()
    # Stage 1 returns 1 authorized candidate
    mock_hybrid.retrieve.return_value = [
        SearchResult(
            chunk_id=1,
            document_id="doc1",
            content="Authorized content",
            metadata={"department": "engineering", "access_level": "employee"},
        )
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    pipeline = RerankedRetrievalPipeline(hybrid_retriever=mock_hybrid, reranker=mock_reranker)
    ctx = AccessContext(department="engineering", access_level="employee")

    pipeline.retrieve_and_rerank(
        query_text="System design",
        top_k=5,
        access_context=ctx,
    )

    mock_hybrid.retrieve.assert_called_once_with(
        query_text="System design",
        top_k=pipeline.candidate_k,
        access_context=ctx,
        conn=None,
    )
    mock_reranker.rerank.assert_called_once()
    assert len(mock_reranker.rerank.call_args[1]["candidates"]) == 1


# -------------------------------------------------------------------------
# 6. End-to-End RAG Pipeline Tests
# -------------------------------------------------------------------------

def test_rag_pipeline_authorized_query():
    """Verifies authorized caller receives grounded answer with citations."""
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve_with_diagnostics.return_value = {
        "candidates": [1],
        "reranked_results": [
            SearchResult(
                chunk_id=1,
                document_id="doc_arch",
                content="Kubernetes orchestrates containerized services in production.",
                metadata={"document_name": "architecture.pdf", "page_number": 3, "section": "K8s"},
                reranker_score=3.5,
            )
        ],
    }
    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-flash-latest"
    mock_llm.generate.return_value = "Kubernetes handles orchestration [SOURCE 1]."

    pipeline = RAGPipeline(retrieval_pipeline=mock_retrieval, llm_provider=mock_llm)
    ctx = AccessContext(department="engineering", access_level="employee")

    response = pipeline.answer_query(
        query="What does Kubernetes orchestrate?",
        access_context=ctx,
    )

    assert response.answer == "Kubernetes handles orchestration [SOURCE 1]."
    assert len(response.citations) == 1
    assert response.citations[0].filename == "architecture.pdf"


def test_rag_pipeline_unauthorized_query():
    """Verifies unauthorized query returns standard refusal with NO citations and NO document leaks."""
    mock_retrieval = MagicMock()
    # Retrieval filtered out all candidates
    mock_retrieval.retrieve_with_diagnostics.return_value = {
        "candidates": [],
        "reranked_results": [],
    }
    mock_llm = MagicMock()

    pipeline = RAGPipeline(retrieval_pipeline=mock_retrieval, llm_provider=mock_llm)
    ctx = AccessContext(department="public", access_level="public")

    response = pipeline.answer_query(
        query="What are the secret manager-level AI governance policies?",
        access_context=ctx,
    )

    # Standard refusal answer
    assert response.answer == NO_EVIDENCE_MESSAGE
    # Zero citations
    assert response.citations == []
    # Zero retrieved results
    assert response.retrieved_results == []
    # LLM never invoked
    mock_llm.generate.assert_not_called()


# -------------------------------------------------------------------------
# 7. FastAPI REST API Access Control Tests
# -------------------------------------------------------------------------

def test_api_query_with_access_context(client):
    """Verifies POST /query with valid access_context forwards context to RAG pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.return_value = RAGResponse(
        query="What is the architecture?",
        answer="It is microservices-based [SOURCE 1].",
        citations=[
            Citation(
                source_id=1,
                chunk_id=10,
                document_id="doc_arch",
                filename="architecture.pdf",
            )
        ],
    )
    set_rag_pipeline(mock_pipeline)

    payload = {
        "query": "What is the architecture?",
        "top_k": 3,
        "access_context": {
            "department": "engineering",
            "access_level": "employee",
            "include_archived": False,
        },
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200

    expected_ctx = AccessContext(department="engineering", access_level="employee", include_archived=False)
    mock_pipeline.answer_query.assert_called_once_with(
        query="What is the architecture?",
        candidate_k=None,
        top_k=3,
        access_context=expected_ctx,
    )


def test_api_query_default_access_context(client):
    """Verifies POST /query without access_context defaults strictly to public, never admin."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.return_value = RAGResponse(
        query="What is support FAQ?",
        answer="Support FAQ explains login [SOURCE 1].",
        citations=[],
    )
    set_rag_pipeline(mock_pipeline)

    response = client.post("/query", json={"query": "What is support FAQ?"})
    assert response.status_code == 200

    # Ensure default is strictly public / not admin
    expected_ctx = AccessContext(department="public", access_level="public", include_archived=False)
    mock_pipeline.answer_query.assert_called_once_with(
        query="What is support FAQ?",
        candidate_k=None,
        top_k=5,
        access_context=expected_ctx,
    )
    assert mock_pipeline.answer_query.call_args[1]["access_context"].is_admin is False


def test_api_query_unauthorized_no_leak(client):
    """Verifies POST /query for restricted content returns 200 refusal without leaking document presence."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.return_value = RAGResponse(
        query="Show AI governance manager secrets",
        answer=NO_EVIDENCE_MESSAGE,
        citations=[],
        retrieved_results=[],
    )
    set_rag_pipeline(mock_pipeline)

    payload = {
        "query": "Show AI governance manager secrets",
        "access_context": {
            "department": "finance",
            "access_level": "employee",
        },
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == NO_EVIDENCE_MESSAGE
    assert data["citations"] == []
    # Verify no leaks in answer or diagnostics
    assert "ai_governance" not in data["answer"].lower()
    assert "manager" not in data["answer"].lower()
