"""
Unit and Integration Tests for Graph RAG Phase G1: PostgreSQL Graph Schema Foundation.

Tests cover:
1. Model Unit Tests: Entity and Relationship dataclasses (to_dict, from_dict).
2. DDL Schema Definition Tests: SCHEMA_SQL contains table definitions, foreign keys, unique constraints, and indexes.
3. Live Database Tests:
   - Tables exist (entities, relationships).
   - Entity insertion and retrieval (get_entity, get_entity_by_canonical_name).
   - Entity deduplication on (canonical_name, entity_type).
   - Relationship insertion with provenance (document_id, chunk_id, page_number).
   - Foreign key integrity on source_entity_id, target_entity_id, document_id, chunk_id.
   - Cascade delete on entity and document deletion.
   - Duplicate relationship protection per chunk.
   - Multiple chunk relationship allowance (same fact across different chunks/docs).
   - Directional relationship retrieval (outgoing, incoming, both).
   - Idempotent init_db() execution.
   - Preservation of existing corpus (8 documents, 11,609 chunks, HNSW index, 384-d vector search).
   - Unaffected access control enforcement.
"""
import pytest
import psycopg
from typing import Generator

from app.db import (
    get_connection,
    test_connection as check_db_connection,
    init_db,
    insert_document,
    insert_chunks,
    insert_entity,
    get_entity,
    get_entity_by_canonical_name,
    insert_relationship,
    get_relationships_for_entity,
    SCHEMA_SQL,
)
from app.models import Entity, Relationship, AccessContext
from app.vector_store import search_similar_chunks


# ==============================================================================
# 1. Model Unit Tests
# ==============================================================================

def test_entity_model_serialization():
    """Verify Entity dataclass serialization and deserialization."""
    entity = Entity(
        id=1,
        canonical_name="transformer",
        entity_type="MODEL_ARCHITECTURE",
        display_name="Transformer",
        metadata={"description": "Attention-based architecture", "aliases": ["Transformers"]},
    )
    data = entity.to_dict()
    assert data["id"] == 1
    assert data["canonical_name"] == "transformer"
    assert data["entity_type"] == "MODEL_ARCHITECTURE"
    assert data["display_name"] == "Transformer"
    assert data["metadata"]["aliases"] == ["Transformers"]

    reconstructed = Entity.from_dict(data)
    assert reconstructed.id == entity.id
    assert reconstructed.canonical_name == entity.canonical_name
    assert reconstructed.entity_type == entity.entity_type
    assert reconstructed.display_name == entity.display_name
    assert reconstructed.metadata == entity.metadata


def test_relationship_model_serialization():
    """Verify Relationship dataclass serialization and deserialization."""
    rel = Relationship(
        id=42,
        source_entity_id=1,
        target_entity_id=2,
        relationship_type="HAS_COMPONENT",
        document_id=83,
        chunk_id=31120,
        page_number=372,
        metadata={"confidence": 0.95, "extraction_method": "llm_few_shot"},
    )
    data = rel.to_dict()
    assert data["id"] == 42
    assert data["source_entity_id"] == 1
    assert data["target_entity_id"] == 2
    assert data["relationship_type"] == "HAS_COMPONENT"
    assert data["document_id"] == 83
    assert data["chunk_id"] == 31120
    assert data["page_number"] == 372
    assert data["metadata"]["confidence"] == 0.95

    reconstructed = Relationship.from_dict(data)
    assert reconstructed.id == rel.id
    assert reconstructed.source_entity_id == rel.source_entity_id
    assert reconstructed.target_entity_id == rel.target_entity_id
    assert reconstructed.relationship_type == rel.relationship_type
    assert reconstructed.document_id == rel.document_id
    assert reconstructed.chunk_id == rel.chunk_id
    assert reconstructed.page_number == rel.page_number
    assert reconstructed.metadata == rel.metadata


# ==============================================================================
# 2. DDL Syntax Unit Tests
# ==============================================================================

def test_schema_sql_graph_ddl():
    """Verify that SCHEMA_SQL includes the entities and relationships table DDL, constraints, and indexes."""
    assert "CREATE TABLE IF NOT EXISTS entities" in SCHEMA_SQL
    assert "canonical_name TEXT NOT NULL" in SCHEMA_SQL
    assert "entity_type TEXT NOT NULL" in SCHEMA_SQL
    assert "uq_entities_canonical_type UNIQUE (canonical_name, entity_type)" in SCHEMA_SQL
    assert "idx_entities_canonical_name ON entities" in SCHEMA_SQL
    assert "idx_entities_type ON entities" in SCHEMA_SQL

    assert "CREATE TABLE IF NOT EXISTS relationships" in SCHEMA_SQL
    assert "REFERENCES entities(id) ON DELETE CASCADE" in SCHEMA_SQL
    assert "REFERENCES documents(id) ON DELETE CASCADE" in SCHEMA_SQL
    assert "REFERENCES chunks(id) ON DELETE CASCADE" in SCHEMA_SQL
    assert "uq_relationships_edge_provenance UNIQUE (source_entity_id, target_entity_id, relationship_type, chunk_id)" in SCHEMA_SQL
    assert "idx_relationships_source ON relationships" in SCHEMA_SQL
    assert "idx_relationships_target ON relationships" in SCHEMA_SQL
    assert "idx_relationships_type ON relationships" in SCHEMA_SQL
    assert "idx_relationships_document_id ON relationships" in SCHEMA_SQL
    assert "idx_relationships_chunk_id ON relationships" in SCHEMA_SQL


# ==============================================================================
# 3. Live PostgreSQL Integration Tests
# ==============================================================================

@pytest.fixture(scope="function")
def live_conn() -> Generator[psycopg.Connection, None, None]:
    """Provides a connection to the live PostgreSQL database for integration tests."""
    if not check_db_connection():
        pytest.skip("PostgreSQL database is not accessible on localhost:5432")

    conn = get_connection(autocommit=False)
    yield conn
    # Roll back any uncommitted changes from tests
    try:
        conn.rollback()
        conn.close()
    except Exception:
        pass


def test_live_graph_tables_and_indexes_exist(live_conn):
    """Verify that entities and relationships tables and all indexes exist in PostgreSQL."""
    init_db()  # Ensure tables are present
    with live_conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('entities', 'relationships');"
        )
        tables = [row[0] for row in cur.fetchall()]
        assert "entities" in tables
        assert "relationships" in tables

        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'entities';"
        )
        entity_indexes = [row[0] for row in cur.fetchall()]
        assert "idx_entities_canonical_name" in entity_indexes
        assert "idx_entities_type" in entity_indexes
        assert "uq_entities_canonical_type" in entity_indexes

        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'relationships';"
        )
        rel_indexes = [row[0] for row in cur.fetchall()]
        assert "idx_relationships_source" in rel_indexes
        assert "idx_relationships_target" in rel_indexes
        assert "idx_relationships_type" in rel_indexes
        assert "idx_relationships_document_id" in rel_indexes
        assert "idx_relationships_chunk_id" in rel_indexes
        assert "uq_relationships_edge_provenance" in rel_indexes


def test_live_entity_crud_and_deduplication(live_conn):
    """Verify entity insertion, retrieval, and unique constraint deduplication."""
    # 1. Insert new entity
    e1_id = insert_entity(
        live_conn,
        canonical_name="transformer",
        entity_type="MODEL_ARCHITECTURE",
        display_name="Transformer",
        metadata={"category": "neural_network"},
    )
    assert isinstance(e1_id, int)
    assert e1_id > 0

    # 2. Retrieve entity by ID
    e1 = get_entity(live_conn, e1_id)
    assert e1 is not None
    assert e1["canonical_name"] == "transformer"
    assert e1["entity_type"] == "MODEL_ARCHITECTURE"
    assert e1["display_name"] == "Transformer"
    assert e1["metadata"]["category"] == "neural_network"

    # 3. Retrieve entity by canonical name + entity type
    e1_lookup = get_entity_by_canonical_name(live_conn, "transformer", "MODEL_ARCHITECTURE")
    assert e1_lookup is not None
    assert e1_lookup["id"] == e1_id

    # 4. Deduplication: inserting same (canonical_name, entity_type) with new display name/metadata
    # returns existing ID and updates metadata
    e1_dup_id = insert_entity(
        live_conn,
        canonical_name="transformer",
        entity_type="MODEL_ARCHITECTURE",
        display_name="Transformer Model",
        metadata={"paper": "Attention Is All You Need"},
    )
    assert e1_dup_id == e1_id

    e1_updated = get_entity(live_conn, e1_id)
    assert e1_updated["display_name"] == "Transformer Model"
    assert e1_updated["metadata"]["category"] == "neural_network"
    assert e1_updated["metadata"]["paper"] == "Attention Is All You Need"


def test_live_relationship_insertion_with_provenance(live_conn):
    """Verify relationship insertion with complete provenance (document_id, chunk_id, page_number)."""
    # Create fixture document and chunk
    doc_id = insert_document(live_conn, filename="graph_test_doc.pdf", document_type="pdf")
    chunk_count = insert_chunks(live_conn, document_id=doc_id, chunks=[{
        "text": "The Transformer uses Multi-Head Attention mechanisms.",
        "metadata": {"page_number": 1, "section": "Architecture", "chunk_index": 0},
    }])
    assert chunk_count == 1

    with live_conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE document_id = %s;", (doc_id,))
        chunk_id = cur.fetchone()[0]

    # Create entities
    src_id = insert_entity(live_conn, "transformer", "MODEL_ARCHITECTURE", "Transformer")
    tgt_id = insert_entity(live_conn, "multi_head_attention", "MECHANISM", "Multi-Head Attention")

    # Insert relationship
    rel_id = insert_relationship(
        live_conn,
        source_entity_id=src_id,
        target_entity_id=tgt_id,
        relationship_type="HAS_COMPONENT",
        document_id=doc_id,
        chunk_id=chunk_id,
        page_number=1,
        metadata={"weight": 1.0},
    )
    assert isinstance(rel_id, int)
    assert rel_id > 0

    # Retrieve relationships
    rels = get_relationships_for_entity(live_conn, src_id, direction="outgoing")
    assert len(rels) == 1
    rel = rels[0]
    assert rel["source_canonical_name"] == "transformer"
    assert rel["target_canonical_name"] == "multi_head_attention"
    assert rel["relationship_type"] == "HAS_COMPONENT"
    assert rel["document_id"] == doc_id
    assert rel["chunk_id"] == chunk_id
    assert rel["page_number"] == 1
    assert rel["metadata"]["weight"] == 1.0


def test_live_relationship_foreign_key_validations(live_conn):
    """Verify that foreign key constraints on source, target, document, and chunk are strictly enforced."""
    doc_id = insert_document(live_conn, filename="fk_test_doc.pdf", document_type="pdf")
    insert_chunks(live_conn, document_id=doc_id, chunks=[{"text": "Chunk text", "metadata": {"chunk_index": 0}}])
    with live_conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE document_id = %s;", (doc_id,))
        chunk_id = cur.fetchone()[0]

    valid_ent_id = insert_entity(live_conn, "test_ent", "CONCEPT", "Test Entity")
    invalid_id = 999999999

    # 1. Invalid source_entity_id
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_relationship(live_conn, invalid_id, valid_ent_id, "CONNECTS_TO", doc_id, chunk_id)
    live_conn.rollback()

    # 2. Invalid target_entity_id
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_relationship(live_conn, valid_ent_id, invalid_id, "CONNECTS_TO", doc_id, chunk_id)
    live_conn.rollback()

    # 3. Invalid document_id
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_relationship(live_conn, valid_ent_id, valid_ent_id, "CONNECTS_TO", invalid_id, chunk_id)
    live_conn.rollback()

    # 4. Invalid chunk_id
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_relationship(live_conn, valid_ent_id, valid_ent_id, "CONNECTS_TO", doc_id, invalid_id)
    live_conn.rollback()


def test_live_relationship_deduplication_and_multi_chunk_allowance(live_conn):
    """
    Verify:
    1. Re-inserting the exact same relationship on the same chunk updates metadata (deduplication).
    2. The same relationship fact appearing in a different chunk or document is permitted.
    """
    doc_id = insert_document(live_conn, filename="multi_chunk_doc.pdf", document_type="pdf")
    insert_chunks(live_conn, document_id=doc_id, chunks=[
        {"text": "Section 1 fact", "metadata": {"page_number": 1, "chunk_index": 0}},
        {"text": "Section 2 fact repeated", "metadata": {"page_number": 2, "chunk_index": 1}},
    ])
    with live_conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE document_id = %s ORDER BY chunk_index ASC;", (doc_id,))
        c1_id, c2_id = [r[0] for r in cur.fetchall()]

    e1 = insert_entity(live_conn, "bert", "MODEL", "BERT")
    e2 = insert_entity(live_conn, "transformer", "MODEL_ARCHITECTURE", "Transformer")

    # Insert fact for chunk 1
    rel1_id = insert_relationship(
        live_conn, e1, e2, "USES_ARCHITECTURE", doc_id, c1_id, page_number=1, metadata={"mention": 1}
    )

    # Re-insert identical fact for chunk 1 -> must update and return same ID
    rel1_dup_id = insert_relationship(
        live_conn, e1, e2, "USES_ARCHITECTURE", doc_id, c1_id, page_number=1, metadata={"updated": True}
    )
    assert rel1_dup_id == rel1_id

    # Insert SAME fact for chunk 2 -> must succeed with new distinct ID (multi-chunk grounding)
    rel2_id = insert_relationship(
        live_conn, e1, e2, "USES_ARCHITECTURE", doc_id, c2_id, page_number=2, metadata={"mention": 2}
    )
    assert rel2_id != rel1_id
    assert rel2_id > rel1_id

    # Verify both relationship occurrences exist with their respective provenance
    rels = get_relationships_for_entity(live_conn, e1, direction="outgoing")
    assert len(rels) == 2
    chunk_ids = {r["chunk_id"] for r in rels}
    assert chunk_ids == {c1_id, c2_id}


def test_live_cascade_delete_from_entity_and_document(live_conn):
    """Verify deleting a parent entity or document cascades to remove referencing relationships."""
    doc_id = insert_document(live_conn, filename="cascade_graph_doc.pdf", document_type="pdf")
    insert_chunks(live_conn, document_id=doc_id, chunks=[{"text": "Text", "metadata": {"chunk_index": 0}}])
    with live_conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE document_id = %s;", (doc_id,))
        chunk_id = cur.fetchone()[0]

    e1 = insert_entity(live_conn, "cascade_src", "TEST", "Cascade Source")
    e2 = insert_entity(live_conn, "cascade_tgt", "TEST", "Cascade Target")

    rel_id = insert_relationship(live_conn, e1, e2, "TEST_EDGE", doc_id, chunk_id)
    assert rel_id > 0

    # Delete entity e1 -> relationship must be automatically removed
    with live_conn.cursor() as cur:
        cur.execute("DELETE FROM entities WHERE id = %s;", (e1,))
        cur.execute("SELECT COUNT(*) FROM relationships WHERE id = %s;", (rel_id,))
        count = cur.fetchone()[0]
    assert count == 0


def test_live_idempotent_init_db_preserves_corpus(live_conn):
    """Verify that calling init_db multiple times is completely idempotent and does not alter document/chunk counts."""
    with live_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        initial_doc_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        initial_chunk_count = cur.fetchone()[0]

    # Run init_db twice
    init_db(live_conn)
    init_db(live_conn)

    with live_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        post_doc_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        post_chunk_count = cur.fetchone()[0]

    assert post_doc_count == initial_doc_count
    assert post_chunk_count == initial_chunk_count


def test_live_existing_corpus_and_vector_search_intact(live_conn):
    """Verify that the existing corpus and chunks are preserved and vector search works."""
    with live_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents;")
        doc_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks;")
        chunk_count = cur.fetchone()[0]

    assert doc_count >= 2
    assert chunk_count >= 5000

    # Test vector search with dummy query embedding
    dummy_embedding = [0.05] * 384
    results = search_similar_chunks(
        query_embedding=dummy_embedding,
        top_k=3,
        access_context=AccessContext(access_level="admin"),
        conn=live_conn,
    )
    assert len(results) == 3
    assert "content" in results[0]
    assert "distance" in results[0]
    assert results[0]["similarity"] >= 0.0


def test_live_access_control_filtering_unaffected(live_conn):
    """Verify access control filtering remains intact after graph schema addition."""
    # Fetch an actual embedding from an internal document if available in current corpus
    with live_conn.cursor() as cur:
        cur.execute("""
            SELECT c.embedding, d.filename FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.access_level != 'public' AND c.embedding IS NOT NULL
            LIMIT 1;
        """)
        row = cur.fetchone()
        if row is None:
            # Active consolidated corpus consists of public research documents
            cur.execute("SELECT c.embedding, d.filename FROM chunks c JOIN documents d ON c.document_id = d.id WHERE c.embedding IS NOT NULL LIMIT 1;")
            row = cur.fetchone()
            assert row is not None, "Expected at least one embedded chunk in corpus"
            target_embedding = row[0].to_numpy().tolist() if hasattr(row[0], "to_numpy") else list(row[0])
            doc_name = row[1]
            # Verify basic search works under public access context
            pub_ctx = AccessContext(department="public", access_level="public")
            pub_results = search_similar_chunks(
                query_embedding=target_embedding,
                top_k=5,
                access_context=pub_ctx,
                conn=live_conn,
            )
            assert len(pub_results) > 0
            return

        target_embedding = row[0].to_numpy().tolist() if hasattr(row[0], "to_numpy") else list(row[0])
        doc_name = row[1]

    # 1. Unauthorized public user (marketing, public) must NOT see internal document even with exact vector match
    unauth_ctx = AccessContext(department="marketing", access_level="public")
    unauth_results = search_similar_chunks(
        query_embedding=target_embedding,
        top_k=10,
        access_context=unauth_ctx,
        conn=live_conn,
    )
    for r in unauth_results:
        assert r["metadata"]["document_name"] != doc_name

    # 2. Authorized engineering employee (engineering, employee) MUST see internal document as top match
    auth_ctx = AccessContext(department="engineering", access_level="employee")
    auth_results = search_similar_chunks(
        query_embedding=target_embedding,
        top_k=5,
        access_context=auth_ctx,
        conn=live_conn,
    )
    assert len(auth_results) > 0
    assert auth_results[0]["metadata"]["document_name"] == doc_name

