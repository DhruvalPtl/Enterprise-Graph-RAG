"""
Unit and Integration Tests for Graph RAG Phase G3: Graph Retrieval Engine.

Tests cover:
1. Model Unit Tests:
   - GraphRetrievalResult dataclass serialization and deserialization (to_dict, from_dict).
   - Empty and total entities properties.
2. Keyword Extraction Tests:
   - Mentions extraction from diverse query styles (proper nouns, quoted strings, acronyms, n-grams).
   - Stop-word filtering and clean handling of empty/whitespace input.
3. Organization & Entity Lookup Tests (Live DB):
   - Finding canonical entities (e.g. 'Stanford University', 'Llama 2 70B') via canonical_name/display_name.
4. Relationship Lookup & Connected Entities Tests (Live DB):
   - Retrieval of semantic edges incident to seed entities with source/target metadata.
5. Provenance Linkage Tests (Live DB):
   - Retrieval and preservation of source chunk IDs, document IDs, and page numbers.
   - Verified chunk content retrieval when include_chunk_content=True.
6. Traversal Depth Tests (depth=1 vs depth=2):
   - Depth 1 retrieves direct incident edges.
   - Depth 2 expands outward to 2-hop neighbor entities.
7. Uncontrolled Graph Traversal Guardrails:
   - Rejection of depth < 1 or depth > 2 with ValueError.
   - Rejection of invalid constructor arguments.
8. Maximum Results Cap Enforcement:
   - Graph expansion strictly bounded by max_results to prevent context explosion.
9. No-Result & Empty Query Handling:
   - Graceful empty results without exceptions for unknown entities and empty strings.
10. Strict Access-Control Filtering:
    - Isolation of confidential/restricted relationships and provenance chunks.
    - Public/cross-department users cannot observe restricted graph relationships.
    - Authorized users and admins can retrieve restricted relationships.
11. Deduplication Verification:
    - Strict deduplication of matched entities, connected entities, relationships, and chunks.
"""
import pytest
import psycopg
from typing import Generator

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
    Entity,
    Relationship,
    GraphRetrievalResult,
)
from app.graph_retriever import GraphRetriever, STOP_WORDS


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def retriever() -> GraphRetriever:
    """Provides a default GraphRetriever instance."""
    return GraphRetriever(default_depth=1, default_max_results=50, default_max_seeds=10)


@pytest.fixture(scope="function")
def live_conn() -> Generator[psycopg.Connection, None, None]:
    """
    Provides a transaction-scoped live connection to PostgreSQL.
    Rolls back any modifications made during the test.
    """
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
# 1. Model Unit Tests
# ==============================================================================

def test_graph_retrieval_result_serialization():
    """Verify GraphRetrievalResult dataclass serialization and deserialization."""
    ent1 = Entity(id=1, canonical_name="llama", entity_type="MODEL", display_name="Llama")
    ent2 = Entity(id=2, canonical_name="meta", entity_type="ORGANIZATION", display_name="Meta")
    rel = Relationship(
        id=10,
        source_entity_id=1,
        target_entity_id=2,
        relationship_type="DEVELOPED_BY",
        document_id=91,
        chunk_id=41000,
        page_number=12,
        metadata={"confidence": 0.98},
    )

    result = GraphRetrievalResult(
        matched_entities=[ent1],
        relationships=[rel],
        connected_entities=[ent2],
        source_chunk_ids=[41000],
        document_ids=[91],
        page_numbers=[12],
        retrieval_metadata={"query": "test query", "status": "success"},
        source_chunks=[{"chunk_id": 41000, "content": "Meta developed Llama."}],
    )

    assert result.is_empty is False
    assert result.total_entities == 2

    data = result.to_dict()
    assert data["source_chunk_ids"] == [41000]
    assert data["document_ids"] == [91]
    assert data["page_numbers"] == [12]
    assert len(data["matched_entities"]) == 1
    assert len(data["relationships"]) == 1
    assert len(data["connected_entities"]) == 1
    assert data["is_empty"] is False
    assert data["total_entities"] == 2

    # Reconstruct
    reconstructed = GraphRetrievalResult.from_dict(data)
    assert reconstructed.source_chunk_ids == [41000]
    assert reconstructed.matched_entities[0].canonical_name == "llama"
    assert reconstructed.relationships[0].relationship_type == "DEVELOPED_BY"
    assert reconstructed.connected_entities[0].canonical_name == "meta"
    assert reconstructed.retrieval_metadata["status"] == "success"


def test_graph_retrieval_result_empty_state():
    """Verify empty result properties and serialization."""
    empty = GraphRetrievalResult()
    assert empty.is_empty is True
    assert empty.total_entities == 0
    assert empty.to_dict()["is_empty"] is True


# ==============================================================================
# 2. Keyword & Candidate Extraction Tests
# ==============================================================================

def test_extract_keywords_multi_style(retriever):
    """Verify candidate extraction handles proper nouns, models, and n-grams."""
    query = 'What did Stanford University and "Llama 2 70B" find regarding the AI Act?'
    keywords = retriever.extract_keywords(query)

    assert "Stanford University" in keywords
    assert "Llama 2 70B" in keywords
    assert "AI Act" in keywords

    # Check that individual common stop words are not present as lone keywords
    assert "what" not in keywords
    assert "did" not in keywords
    assert "and" not in keywords
    assert "the" not in keywords


def test_extract_keywords_empty_or_whitespace(retriever):
    """Verify empty and whitespace-only queries return empty candidates list."""
    assert retriever.extract_keywords("") == []
    assert retriever.extract_keywords("   \n\t  ") == []


# ==============================================================================
# 3. Organization & Entity Lookup Tests (Live PostgreSQL DB)
# ==============================================================================

def test_organization_entity_lookup_live(retriever):
    """Verify looking up an organization entity (Stanford University) from live data."""
    res = retriever.retrieve("Stanford University", depth=1, max_results=10)

    assert not res.is_empty
    assert len(res.matched_entities) > 0
    assert any("stanford" in e.canonical_name.lower() for e in res.matched_entities)
    assert len(res.relationships) > 0
    assert len(res.connected_entities) > 0
    assert res.retrieval_metadata["status"] == "success"


def test_model_entity_lookup_live(retriever):
    """Verify looking up a model entity (Llama 2 70B or Gemini Ultra)."""
    res = retriever.retrieve("What benchmarks were measured for Gemini Ultra?", depth=1, max_results=10)

    assert not res.is_empty
    matched_names = [e.canonical_name for e in res.matched_entities]
    assert any("Gemini Ultra" in name for name in matched_names)
    assert len(res.relationships) > 0
    assert res.retrieval_metadata["seed_count"] > 0


# ==============================================================================
# 4. Relationship Lookup & Connected Entities Tests (Live DB)
# ==============================================================================

def test_relationship_lookup_live(retriever):
    """Verify relationship attributes and connected entities."""
    res = retriever.retrieve("Gemini Ultra", depth=1, max_results=15)

    assert len(res.relationships) > 0
    first_rel = res.relationships[0]
    assert first_rel.id is not None
    assert first_rel.source_entity_id > 0
    assert first_rel.target_entity_id > 0
    assert isinstance(first_rel.relationship_type, str) and len(first_rel.relationship_type) > 0

    # Ensure source_name and target_name are populated in metadata
    assert "source_name" in first_rel.metadata
    assert "target_name" in first_rel.metadata

    # Ensure connected entities are populated and distinct from matched seeds
    assert len(res.connected_entities) > 0
    seed_ids = {e.id for e in res.matched_entities}
    for ce in res.connected_entities:
        assert ce.id not in seed_ids


# ==============================================================================
# 5. Provenance Linkage Tests (Live DB)
# ==============================================================================

def test_provenance_linkage_live(retriever):
    """Verify preservation of chunk IDs, document IDs, and page numbers."""
    res = retriever.retrieve("Stanford University", depth=1, max_results=10)

    assert len(res.source_chunk_ids) > 0
    assert all(isinstance(cid, int) and cid > 0 for cid in res.source_chunk_ids)
    assert len(res.document_ids) > 0
    assert all(isinstance(did, int) and did > 0 for did in res.document_ids)
    assert len(res.page_numbers) > 0
    assert all(isinstance(p, int) and p >= 0 for p in res.page_numbers)


def test_provenance_with_chunk_content_live(retriever):
    """Verify chunk text and metadata are fetched when include_chunk_content=True."""
    res = retriever.retrieve("Stanford University", depth=1, max_results=5, include_chunk_content=True)

    assert len(res.source_chunks) > 0
    chunk_sample = res.source_chunks[0]
    assert "chunk_id" in chunk_sample
    assert "document_id" in chunk_sample
    assert "content" in chunk_sample
    assert isinstance(chunk_sample["content"], str) and len(chunk_sample["content"]) > 0
    assert "page_number" in chunk_sample


# ==============================================================================
# 6. Traversal Depth Tests (depth=1 vs depth=2)
# ==============================================================================

def test_traversal_depth_expansion_live(retriever):
    """
    Verify that depth=2 expands traversal beyond depth=1.
    Uses 'UniAudio' which has a moderate degree at depth 1.
    """
    res_d1 = retriever.retrieve("UniAudio", depth=1, max_results=50)
    res_d2 = retriever.retrieve("UniAudio", depth=2, max_results=50)

    assert not res_d1.is_empty
    assert not res_d2.is_empty
    # Depth 2 should have equal or greater relationships and connected entities
    assert len(res_d2.relationships) >= len(res_d1.relationships)
    assert len(res_d2.connected_entities) >= len(res_d1.connected_entities)
    assert res_d1.retrieval_metadata["depth"] == 1
    assert res_d2.retrieval_metadata["depth"] == 2


# ==============================================================================
# 7. Uncontrolled Graph Traversal Guardrails
# ==============================================================================

def test_uncontrolled_traversal_prevention(retriever):
    """Verify that depths outside (1, 2) raise ValueError to prevent runaway queries."""
    with pytest.raises(ValueError, match="Traversal depth must be 1 or 2"):
        retriever.retrieve("Stanford University", depth=0)

    with pytest.raises(ValueError, match="Traversal depth must be 1 or 2"):
        retriever.retrieve("Stanford University", depth=3)

    with pytest.raises(ValueError, match="Traversal depth must be 1 or 2"):
        retriever.retrieve("Stanford University", depth=-1)

    with pytest.raises(ValueError, match="Traversal depth must be 1 or 2"):
        GraphRetriever(default_depth=5)


# ==============================================================================
# 8. Maximum Results Cap Enforcement
# ==============================================================================

def test_max_results_cap_enforcement(retriever):
    """Verify that the relationship count never exceeds the configured max_results."""
    for cap in (3, 7, 15):
        res = retriever.retrieve("Stanford University", depth=1, max_results=cap)
        assert len(res.relationships) <= cap
        assert res.retrieval_metadata["max_results"] == cap


# ==============================================================================
# 9. No-Result & Empty Query Handling
# ==============================================================================

def test_no_result_query_live(retriever):
    """Verify unknown queries return a clean, empty result gracefully."""
    res = retriever.retrieve("NonExistentEntityXyz999888777", depth=1)

    assert res.is_empty
    assert len(res.matched_entities) == 0
    assert len(res.relationships) == 0
    assert len(res.connected_entities) == 0
    assert len(res.source_chunk_ids) == 0
    assert res.retrieval_metadata["status"] == "no_seed_entities_found"


def test_empty_string_query(retriever):
    """Verify empty or whitespace query returns clean empty result."""
    res = retriever.retrieve("   ", depth=1)

    assert res.is_empty
    assert res.retrieval_metadata["status"] == "empty_query"


# ==============================================================================
# 10. Strict Access-Control Filtering Tests
# ==============================================================================

def test_access_control_confidential_document_isolation(live_conn):
    """
    Verify that relationships linked to restricted documents are strictly filtered out
    for unauthorized users, while remaining accessible to authorized users and admins.
    """
    # 1. Insert a confidential document with manager-level engineering access
    doc_id = insert_document(
        conn=live_conn,
        filename="confidential_rnd_report.pdf",
        department="engineering",
        access_level="manager",
        status="active",
    )

    # 2. Insert entities and relationship pointing to this confidential document
    e1 = insert_entity(live_conn, "UniqueSecretProjectAlpha", "PROJECT", "Project Alpha")
    e2 = insert_entity(live_conn, "UniqueSecretModelBeta", "MODEL", "Model Beta")
    rel_id = insert_relationship(
        conn=live_conn,
        source_entity_id=e1,
        target_entity_id=e2,
        relationship_type="POWERED_BY",
        document_id=doc_id,
        chunk_id=None,
        page_number=42,
    )

    retriever = GraphRetriever()

    # Case A: Public user (unauthorized)
    public_ctx = AccessContext(department="public", access_level="public")
    res_public = retriever.retrieve(
        query="UniqueSecretProjectAlpha",
        access_context=public_ctx,
        conn=live_conn,
    )
    # The relationship and provenance must NOT be visible
    assert len(res_public.relationships) == 0
    assert len(res_public.connected_entities) == 0
    assert doc_id not in res_public.document_ids

    # Case B: Engineering employee (under-privileged access level)
    emp_ctx = AccessContext(department="engineering", access_level="employee")
    res_emp = retriever.retrieve(
        query="UniqueSecretProjectAlpha",
        access_context=emp_ctx,
        conn=live_conn,
    )
    assert len(res_emp.relationships) == 0
    assert doc_id not in res_emp.document_ids

    # Case C: Marketing manager (wrong department)
    mkt_ctx = AccessContext(department="marketing", access_level="manager")
    res_mkt = retriever.retrieve(
        query="UniqueSecretProjectAlpha",
        access_context=mkt_ctx,
        conn=live_conn,
    )
    assert len(res_mkt.relationships) == 0
    assert doc_id not in res_mkt.document_ids

    # Case D: Engineering manager (AUTHORIZED)
    eng_mgr_ctx = AccessContext(department="engineering", access_level="manager")
    res_mgr = retriever.retrieve(
        query="UniqueSecretProjectAlpha",
        access_context=eng_mgr_ctx,
        conn=live_conn,
    )
    assert len(res_mgr.relationships) == 1
    assert res_mgr.relationships[0].relationship_type == "POWERED_BY"
    assert len(res_mgr.connected_entities) == 1
    assert res_mgr.connected_entities[0].canonical_name == "UniqueSecretModelBeta"
    assert doc_id in res_mgr.document_ids

    # Case E: Admin user (UNIVERSAL ACCESS)
    admin_ctx = AccessContext(department="any", access_level="admin")
    res_admin = retriever.retrieve(
        query="UniqueSecretProjectAlpha",
        access_context=admin_ctx,
        conn=live_conn,
    )
    assert len(res_admin.relationships) == 1
    assert doc_id in res_admin.document_ids


def test_access_control_archived_document_isolation(live_conn):
    """Verify that relationships in archived documents are omitted unless include_archived=True."""
    doc_id = insert_document(
        conn=live_conn,
        filename="deprecated_architecture.pdf",
        department="public",
        access_level="public",
        status="archived",
    )
    e1 = insert_entity(live_conn, "UniqueLegacyArch99", "ARCHITECTURE", "Legacy Arch 99")
    e2 = insert_entity(live_conn, "UniqueLegacyFramework99", "FRAMEWORK", "Legacy Framework 99")
    insert_relationship(
        conn=live_conn,
        source_entity_id=e1,
        target_entity_id=e2,
        relationship_type="SUPERSEDES",
        document_id=doc_id,
    )

    retriever = GraphRetriever()

    # Standard query without archived docs
    res_active = retriever.retrieve(
        query="UniqueLegacyArch99",
        access_context=AccessContext(include_archived=False),
        conn=live_conn,
    )
    assert len(res_active.relationships) == 0
    assert doc_id not in res_active.document_ids

    # Query with include_archived=True
    res_archived = retriever.retrieve(
        query="UniqueLegacyArch99",
        access_context=AccessContext(include_archived=True),
        conn=live_conn,
    )
    assert len(res_archived.relationships) == 1
    assert doc_id in res_archived.document_ids


# ==============================================================================
# 11. Deduplication Tests
# ==============================================================================

def test_deduplication_of_nodes_and_provenance(retriever):
    """Verify entities, relationships, chunk IDs, and document IDs are deduplicated."""
    res = retriever.retrieve("Stanford University", depth=2, max_results=30)

    # 1. Matched entity IDs are unique
    matched_ids = [e.id for e in res.matched_entities]
    assert len(matched_ids) == len(set(matched_ids))

    # 2. Connected entity IDs are unique and disjoint from matched entity IDs
    connected_ids = [e.id for e in res.connected_entities]
    assert len(connected_ids) == len(set(connected_ids))
    assert set(matched_ids).isdisjoint(set(connected_ids))

    # 3. Relationship IDs are unique
    rel_ids = [r.id for r in res.relationships]
    assert len(rel_ids) == len(set(rel_ids))

    # 4. Chunk IDs and Document IDs are strictly unique
    assert len(res.source_chunk_ids) == len(set(res.source_chunk_ids))
    assert len(res.document_ids) == len(set(res.document_ids))
    assert len(res.page_numbers) == len(set(res.page_numbers))
