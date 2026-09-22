"""
Unit and Integration Tests for Graph + Vector Hybrid RAG (Phase G4).

Tests cover:
1. Dual-Path Retrieval: Vector + Graph results are both retrieved for a user query.
2. Vector-Only Fallback: When graph branch finds no entities, vector candidates are used.
3. Graph-Only Fallback: When vector branch yields no candidates, graph candidates are used.
4. Empty Branches: When neither branch returns candidates, graceful refusal is returned.
5. Deduplication: Identical chunk IDs returned by both branches are merged into a single candidate.
6. Provenance Tracking: Fused chunks record retrieved_by = ["vector", "graph"], ["vector"], or ["graph"].
7. Graph-Only Provenance: Uniquely graph-derived chunks preserve their graph provenance tags.
8. Real Source Chunk Linkage: Graph relationships remain mapped to real source chunks with content.
9. Cross-Encoder Reranking: Fused candidates are scored uniformly by the cross-encoder.
10. Final Top-K Respect: Output reranked results strictly observe the requested top_k limit.
11. Multi-Tenant Access Control: Multi-level authorization (Public, Employee, Manager, Admin)
    is preserved across both branches and validated during fusion.
12. Unauthorized Evidence Leak Prevention: Confidential relationships and chunks never enter fused results, context, or citations.
13. Archived Document Exclusion: Archived documents remain inaccessible unless include_archived=True.
14. Context Builder & Citation Generation: Citations point to authentic documents and pages; compact [GRAPH RELATIONSHIP] headers are included.
15. End-to-End RAGPipeline Integration: RAGPipeline executes hybrid retrieval and produces grounded answers.
"""
import pytest
import psycopg
from typing import Generator, List, Dict, Any, Optional

from app.db import (
    get_connection,
    test_connection as check_db_connection,
    insert_document,
    insert_entity,
    insert_relationship,
    insert_chunks,
)
from app.models import (
    AccessContext,
    SearchResult,
    Citation,
    RAGResponse,
    Entity,
    Relationship,
    GraphRetrievalResult,
    HybridRetrievalResult,
)
from app.reranker import CrossEncoderReranker, RerankedRetrievalPipeline
from app.graph_retriever import GraphRetriever
from app.hybrid_retriever import GraphVectorHybridRetriever
from app.context_builder import ContextBuilder
from app.rag import RAGPipeline
from app.llm import GeminiLLMProvider


# ==============================================================================
# Mocks & Test Doubles for Unit Testing
# ==============================================================================

class MockRerankedPipeline:
    """Mock RerankedRetrievalPipeline that returns controlled vector candidates."""
    def __init__(self, candidates: List[Dict[str, Any]]):
        self.candidates = candidates

    @property
    def hybrid_retriever(self):
        class _Inner:
            def __init__(self, cands):
                self.cands = cands
            def retrieve_with_details(self, *args, **kwargs):
                return {"fused_results": self.cands}
        return _Inner(self.candidates)


class MockGraphRetriever:
    """Mock GraphRetriever that returns controlled graph retrieval results."""
    def __init__(self, result: Optional[GraphRetrievalResult] = None):
        self.result = result or GraphRetrievalResult()

    def retrieve(self, *args, **kwargs) -> GraphRetrievalResult:
        return self.result


class MockCrossEncoder:
    """Mock CrossEncoder that assigns predetermined scores based on chunk_id."""
    def __init__(self, score_map: Optional[Dict[Any, float]] = None):
        self.score_map = score_map or {}

    def predict(self, pairs: List[List[str]], batch_size: int = 32):
        # Return predictable descending scores
        return [self.score_map.get(idx, 10.0 - idx) for idx in range(len(pairs))]


class MockLLM:
    """Mock LLM returning deterministic grounded answers."""
    def __init__(self, answer: str = "According to [SOURCE 1], Gemini Ultra scored high on MMLU."):
        self.model_name = "mock-gemini-pro"
        self._answer = answer

    def generate(self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.0) -> str:
        return self._answer


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="function")
def live_conn() -> Generator[psycopg.Connection, None, None]:
    """Provides a transaction-scoped live connection to PostgreSQL."""
    if not check_db_connection():
        pytest.skip("PostgreSQL database is not accessible on localhost:5432")

    conn = get_connection(autocommit=False)
    yield conn
    try:
        conn.rollback()
        conn.close()
    except Exception:
        pass


# ==============================================================================
# 1. Unit Tests: Fusion, Deduplication, Provenance Attribution
# ==============================================================================

def test_evidence_fusion_and_deduplication():
    """Verify duplicate chunks from vector and graph are merged and provenance tracked."""
    # Chunk 101 returned by vector
    # Chunk 102 returned by vector
    vector_candidates = [
        {"chunk_id": 101, "document_id": 1, "content": "Chunk 101 text", "metadata": {"document_name": "doc1.pdf", "page_number": 1, "department": "public", "access_level": "public", "status": "active"}},
        {"chunk_id": 102, "document_id": 1, "content": "Chunk 102 text", "metadata": {"document_name": "doc1.pdf", "page_number": 2, "department": "public", "access_level": "public", "status": "active"}},
    ]

    # Chunk 101 returned by graph (OVERLAP)
    # Chunk 103 returned by graph only (GRAPH ONLY)
    graph_res = GraphRetrievalResult(
        matched_entities=[Entity(id=1, canonical_name="gemini", entity_type="MODEL", display_name="Gemini")],
        relationships=[
            Relationship(id=50, source_entity_id=1, target_entity_id=2, relationship_type="EVALUATED_ON", document_id=1, chunk_id=101, page_number=1, metadata={"source_name": "Gemini", "target_name": "MMLU"}),
            Relationship(id=51, source_entity_id=1, target_entity_id=3, relationship_type="DEVELOPED_BY", document_id=1, chunk_id=103, page_number=3, metadata={"source_name": "Gemini", "target_name": "Google"}),
        ],
        source_chunk_ids=[101, 103],
        document_ids=[1],
        page_numbers=[1, 3],
        source_chunks=[
            {"chunk_id": 101, "document_id": 1, "page_number": 1, "content": "Chunk 101 text", "document_name": "doc1.pdf", "department": "public", "access_level": "public", "status": "active"},
            {"chunk_id": 103, "document_id": 1, "page_number": 3, "content": "Chunk 103 text", "document_name": "doc1.pdf", "department": "public", "access_level": "public", "status": "active"},
        ],
    )

    reranker = CrossEncoderReranker(model=MockCrossEncoder())
    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(vector_candidates),
        graph_retriever=MockGraphRetriever(graph_res),
        reranker=reranker,
    )

    result = hybrid.retrieve_fused("test query", top_k=5)

    # 1. Total unique fused candidates should be 3 (101, 102, 103)
    assert len(result.fused_candidates) == 3
    assert result.deduplication_stats["vector_candidates"] == 2
    assert result.deduplication_stats["graph_candidates"] == 2
    assert result.deduplication_stats["overlap_count"] == 1
    assert result.deduplication_stats["fused_count"] == 3

    # Find candidates by chunk_id
    cand_map = {c.chunk_id: c for c in result.fused_candidates}

    # 2. Overlapped Chunk 101 must record retrieved_by = ["vector", "graph"]
    assert set(cand_map[101].sources["retrieved_by"]) == {"vector", "graph"}
    assert cand_map[101].source == "hybrid_graph_vector"
    assert "graph_relationships" in cand_map[101].metadata
    assert cand_map[101].metadata["graph_relationships"][0]["relationship_type"] == "EVALUATED_ON"

    # 3. Vector-only Chunk 102
    assert cand_map[102].sources["retrieved_by"] == ["vector"]

    # 4. Graph-only Chunk 103
    assert cand_map[103].sources["retrieved_by"] == ["graph"]
    assert cand_map[103].source == "graph"


def test_vector_only_fallback():
    """Verify that when graph retrieval returns nothing, vector candidates work normally."""
    vector_candidates = [
        {"chunk_id": 201, "document_id": 1, "content": "Vector candidate", "metadata": {"document_name": "doc1.pdf", "department": "public", "access_level": "public", "status": "active"}},
    ]
    empty_graph = GraphRetrievalResult()

    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(vector_candidates),
        graph_retriever=MockGraphRetriever(empty_graph),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    result = hybrid.retrieve_fused("test query", top_k=5)

    assert len(result.fused_candidates) == 1
    assert len(result.reranked_results) == 1
    assert result.reranked_results[0].chunk_id == 201
    assert result.reranked_results[0].sources["retrieved_by"] == ["vector"]
    assert result.retrieval_metadata["graph_seed_count"] == 0


def test_graph_only_fallback():
    """Verify that when vector retrieval returns nothing, graph candidates work normally."""
    empty_vector = []
    graph_res = GraphRetrievalResult(
        matched_entities=[Entity(id=5, canonical_name="stanford", entity_type="ORGANIZATION", display_name="Stanford")],
        relationships=[Relationship(id=80, source_entity_id=5, target_entity_id=6, relationship_type="AFFILIATED_WITH", document_id=1, chunk_id=301, page_number=5, metadata={"source_name": "Stanford", "target_name": "HAI"})],
        source_chunk_ids=[301],
        document_ids=[1],
        page_numbers=[5],
        source_chunks=[
            {"chunk_id": 301, "document_id": 1, "page_number": 5, "content": "Stanford HAI graph chunk", "document_name": "ai_index.pdf", "department": "public", "access_level": "public", "status": "active"}
        ],
    )

    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(empty_vector),
        graph_retriever=MockGraphRetriever(graph_res),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    result = hybrid.retrieve_fused("test query", top_k=5)

    assert len(result.fused_candidates) == 1
    assert len(result.reranked_results) == 1
    assert result.reranked_results[0].chunk_id == 301
    assert result.fused_candidates[0].source == "graph"
    assert result.reranked_results[0].sources["retrieved_by"] == ["graph"]


def test_both_branches_empty():
    """Verify that when both branches return no candidates, clean empty result is returned."""
    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline([]),
        graph_retriever=MockGraphRetriever(GraphRetrievalResult()),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    result = hybrid.retrieve_fused("test query", top_k=5)

    assert result.is_empty
    assert len(result.fused_candidates) == 0
    assert len(result.reranked_results) == 0
    assert result.retrieval_metadata["status"] == "no_evidence"


def test_final_top_k_respected():
    """Verify that reranked output strictly respects top_k cap."""
    cands = [
        {"chunk_id": i, "document_id": 1, "content": f"Content {i}", "metadata": {"document_name": "d.pdf", "department": "public", "access_level": "public", "status": "active"}}
        for i in range(10)
    ]
    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(cands),
        graph_retriever=MockGraphRetriever(GraphRetrievalResult()),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    for k in (1, 3, 5):
        res = hybrid.retrieve_fused("query", top_k=k)
        assert len(res.reranked_results) == k


# ==============================================================================
# 2. Access Control & Authorization Tests (G4.9)
# ==============================================================================

def test_access_control_filtering_during_fusion():
    """Verify unauthorized candidates from either branch are omitted at the fusion boundary."""
    # Chunk 401 is public (authorized for anyone)
    # Chunk 402 is confidential / manager engineering
    vector_candidates = [
        {"chunk_id": 401, "document_id": 1, "content": "Public text", "metadata": {"document_name": "public.pdf", "department": "public", "access_level": "public", "status": "active"}},
        {"chunk_id": 402, "document_id": 2, "content": "Restricted text", "metadata": {"document_name": "secret.pdf", "department": "engineering", "access_level": "manager", "status": "active"}},
    ]

    # Graph candidate with manager engineering access
    graph_res = GraphRetrievalResult(
        matched_entities=[Entity(id=1, canonical_name="secret", entity_type="PROJECT", display_name="Secret")],
        relationships=[Relationship(id=1, source_entity_id=1, target_entity_id=2, relationship_type="USES", document_id=2, chunk_id=403)],
        source_chunk_ids=[403],
        document_ids=[2],
        source_chunks=[
            {"chunk_id": 403, "document_id": 2, "content": "Secret graph text", "document_name": "secret.pdf", "department": "engineering", "access_level": "manager", "status": "active"}
        ],
    )

    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(vector_candidates),
        graph_retriever=MockGraphRetriever(graph_res),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    # Public user: should only see chunk 401; chunks 402 and 403 MUST be excluded
    pub_ctx = AccessContext(department="public", access_level="public")
    res_pub = hybrid.retrieve_fused("query", access_context=pub_ctx)
    assert len(res_pub.fused_candidates) == 1
    assert res_pub.fused_candidates[0].chunk_id == 401

    # Engineering Employee: under-privileged, still cannot see manager chunks
    emp_ctx = AccessContext(department="engineering", access_level="employee")
    res_emp = hybrid.retrieve_fused("query", access_context=emp_ctx)
    assert len(res_emp.fused_candidates) == 1
    assert res_emp.fused_candidates[0].chunk_id == 401

    # Engineering Manager: AUTHORIZED, should see all 3 chunks
    mgr_ctx = AccessContext(department="engineering", access_level="manager")
    res_mgr = hybrid.retrieve_fused("query", access_context=mgr_ctx)
    assert len(res_mgr.fused_candidates) == 3
    assert {c.chunk_id for c in res_mgr.fused_candidates} == {401, 402, 403}

    # Admin: Universal access, sees all 3 chunks
    admin_ctx = AccessContext(department="any", access_level="admin")
    res_admin = hybrid.retrieve_fused("query", access_context=admin_ctx)
    assert len(res_admin.fused_candidates) == 3


def test_access_control_archived_documents_isolation():
    """Verify archived candidates are excluded unless include_archived=True."""
    cands = [
        {"chunk_id": 501, "document_id": 1, "content": "Archived text", "metadata": {"document_name": "old.pdf", "department": "public", "access_level": "public", "status": "archived"}},
    ]
    hybrid = GraphVectorHybridRetriever(
        reranked_pipeline=MockRerankedPipeline(cands),
        graph_retriever=MockGraphRetriever(GraphRetrievalResult()),
        reranker=CrossEncoderReranker(model=MockCrossEncoder()),
    )

    # Standard context (include_archived=False)
    res_active = hybrid.retrieve_fused("query", access_context=AccessContext(include_archived=False))
    assert len(res_active.fused_candidates) == 0

    # With include_archived=True
    res_archived = hybrid.retrieve_fused("query", access_context=AccessContext(include_archived=True))
    assert len(res_archived.fused_candidates) == 1


# ==============================================================================
# 3. Context Builder & Citation Formatting Tests (G4.7 & G4.8)
# ==============================================================================

def test_context_builder_graph_relationship_header():
    """Verify ContextBuilder includes compact [GRAPH RELATIONSHIP] header for graph chunks."""
    result_with_graph = SearchResult(
        chunk_id=41101,
        document_id=91,
        content="Gemini Ultra became the first LLM to reach humanlevel performance on MMLU.",
        metadata={
            "document_name": "AI_Index_2024.pdf",
            "page_number": 3,
            "section": "Technical Progress",
            "graph_relationships": [
                {"source": "Gemini Ultra", "relationship_type": "EVALUATED_ON", "target": "MMLU"}
            ],
        },
        source="hybrid_graph_vector",
        sources={"retrieved_by": ["vector", "graph"]},
    )

    builder = ContextBuilder(max_context_chars=2000)
    built = builder.build_context("query", results=[result_with_graph])

    assert "[GRAPH RELATIONSHIP] Gemini Ultra --[EVALUATED_ON]--> MMLU" in built.context_text
    assert "Document: AI_Index_2024.pdf" in built.context_text
    assert "Page: 3" in built.context_text
    assert len(built.citations) == 1

    citation = built.citations[0]
    assert citation.chunk_id == 41101
    assert citation.filename == "AI_Index_2024.pdf"
    assert citation.page_number == 3
    assert citation.sources.get("retrieved_by") == ["vector", "graph"]


# ==============================================================================
# 4. End-to-End RAGPipeline Integration Tests
# ==============================================================================

def test_rag_pipeline_hybrid_integration():
    """Verify RAGPipeline orchestrates hybrid retrieval, context building, and LLM response."""
    # Chunk with graph evidence
    cand = SearchResult(
        chunk_id=41101,
        document_id=91,
        content="Gemini Ultra reached 90% on MMLU benchmark.",
        metadata={"document_name": "report.pdf", "page_number": 3, "department": "public", "access_level": "public", "status": "active"},
        score=4.5,
        source="hybrid_graph_vector",
        sources={"retrieved_by": ["vector", "graph"]},
    )

    # Hybrid retriever mock
    class MockHybridRetriever:
        def retrieve_fused(self, query, **kwargs):
            return HybridRetrievalResult(
                query=query,
                reranked_results=[cand],
                fused_candidates=[cand],
                vector_candidates=[cand],
                graph_candidates=[cand],
                deduplication_stats={"overlap_count": 1, "fused_count": 1},
                retrieval_metadata={"retrieval_mode": "hybrid_graph_vector", "graph_seed_count": 1, "graph_relationship_count": 1},
            )

    pipeline = RAGPipeline(
        retrieval_pipeline=MockHybridRetriever(),
        llm_provider=MockLLM(),
        enable_graph=True,
    )

    response = pipeline.answer_query("What benchmark was Gemini Ultra evaluated on?")

    assert isinstance(response, RAGResponse)
    assert "Gemini Ultra" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == 41101
    assert response.diagnostics.get("retrieval_mode") == "hybrid_graph_vector"
    assert response.diagnostics.get("graph_seed_count") == 1
    assert response.diagnostics.get("graph_relationship_count") == 1


def test_rag_pipeline_no_evidence_behavior():
    """Verify RAGPipeline returns standard refusal when hybrid retrieval finds no candidates."""
    class EmptyHybridRetriever:
        def retrieve_fused(self, query, **kwargs):
            return HybridRetrievalResult(query=query)

    pipeline = RAGPipeline(
        retrieval_pipeline=EmptyHybridRetriever(),
        llm_provider=MockLLM(),
        enable_graph=True,
    )

    response = pipeline.answer_query("Unknown question")

    assert response.citations == []
    assert response.retrieved_results == []
    assert "No candidate chunks found during retrieval" in response.diagnostics.get("reason", "")


# ==============================================================================
# 5. Live PostgreSQL Integration Tests
# ==============================================================================

def test_live_graph_and_vector_dual_retrieval(live_conn):
    """
    Exercise real PostgreSQL graph data and vector store for an entity with explicit relationships.
    Query: 'What organization developed Gemini Ultra and what benchmark was it evaluated on?'
    """
    retriever = GraphVectorHybridRetriever()
    query = "What organization developed Gemini Ultra and what benchmark was it evaluated on?"

    result = retriever.retrieve_fused(
        query=query,
        top_k=5,
        access_context=AccessContext(),
        conn=live_conn,
    )

    assert not result.is_empty
    assert len(result.fused_candidates) > 0
    assert len(result.reranked_results) > 0

    # Ensure both vector and graph branches executed
    assert result.retrieval_metadata["retrieval_mode"] == "hybrid_graph_vector"
    assert result.deduplication_stats["vector_candidates"] > 0
    assert result.deduplication_stats["graph_candidates"] > 0

    # Verify at least one result has graph provenance
    has_graph_provenance = any(
        "graph" in r.sources.get("retrieved_by", [])
        for r in result.reranked_results
    )
    assert has_graph_provenance, "Expected at least one top reranked result with graph provenance"

    # Verify real chunk IDs and document IDs are positive integers
    for r in result.reranked_results:
        assert isinstance(r.chunk_id, int) and r.chunk_id > 0
        assert isinstance(r.document_id, int) and r.document_id > 0
        assert len(r.content) > 0


def test_live_access_control_isolation(live_conn):
    """
    Verify on live DB that restricted graph relationships and their provenance chunks
    are strictly isolated between Public and Manager contexts.
    """
    # 1. Insert a confidential document with manager clearance
    doc_id = insert_document(
        conn=live_conn,
        filename="confidential_rnd_strategy.pdf",
        department="engineering",
        access_level="manager",
        status="active",
    )

    # 2. Insert entities and relationship
    e1 = insert_entity(live_conn, "UniqueSecretProjectGamma", "PROJECT", "Project Gamma")
    e2 = insert_entity(live_conn, "UniqueSecretBenchmarkDelta", "BENCHMARK", "Benchmark Delta")
    insert_relationship(
        conn=live_conn,
        source_entity_id=e1,
        target_entity_id=e2,
        relationship_type="EVALUATED_ON",
        document_id=doc_id,
        page_number=99,
    )

    retriever = GraphVectorHybridRetriever()

    # Public user: must NOT retrieve the secret project or benchmark
    pub_ctx = AccessContext(department="public", access_level="public")
    res_pub = retriever.retrieve_fused("UniqueSecretProjectGamma", access_context=pub_ctx, conn=live_conn)
    for cand in res_pub.fused_candidates:
        assert cand.document_id != doc_id

    # Engineering Manager: AUTHORIZED, must be able to retrieve the secret project
    mgr_ctx = AccessContext(department="engineering", access_level="manager")
    res_mgr = retriever.retrieve_fused("UniqueSecretProjectGamma", access_context=mgr_ctx, conn=live_conn)
    # The confidential relationship was found in the graph
    assert res_mgr.graph_result is not None
    assert any(r.document_id == doc_id for r in res_mgr.graph_result.relationships)
