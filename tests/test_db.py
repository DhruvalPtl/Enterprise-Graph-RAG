"""
Unit and Integration tests for PostgreSQL Database Layer.

Tests cover:
- Database configuration parsing
- Connection error handling & security (credentials not leaked in errors)
- Schema DDL definitions & foreign key constraints
- Mocked execution of document & chunk insert logic
- Live PostgreSQL integration tests (automatically skipped if no PostgreSQL instance is running on localhost:5432)
"""
import pytest
from unittest.mock import MagicMock, patch
import psycopg

from app.db import (
    get_db_config,
    get_connection,
    test_connection as check_db_connection,
    init_db,
    insert_document,
    insert_chunks,
    SCHEMA_SQL,
)


# --- 1. Unit Tests (Always Run Offline) ---

def test_check_db_connection_returns_bool():
    """Verifies that check_db_connection safely returns a boolean without throwing unhandled exceptions."""
    result = check_db_connection()
    assert isinstance(result, bool)


def test_db_config_defaults():
    """Verifies that database configuration properly loads host, port, db, and user."""
    config = get_db_config()
    assert "host" in config or "conninfo" in config
    if "host" in config:
        assert config["port"] == 5432
        assert config["dbname"] == "rag_db"
        assert config["user"] == "postgres"


def test_connection_error_handling_when_offline():
    """Verifies clear ConnectionError is raised when DB is unreachable, without leaking passwords."""
    with patch("app.db.POSTGRES_PORT", 59999):  # Unused port
        with pytest.raises(ConnectionError) as exc_info:
            get_connection()
        msg = str(exc_info.value)
        assert "Could not connect to PostgreSQL" in msg
        # Passwords must never appear in error messages
        assert "your_postgres_password_here" not in msg


def test_schema_sql_ddl_syntax():
    """Verifies that the DDL string defines vector extension, both tables, foreign key, and HNSW index."""
    assert "CREATE EXTENSION IF NOT EXISTS vector" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS documents" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS chunks" in SCHEMA_SQL
    assert "VECTOR(384)" in SCHEMA_SQL
    assert "REFERENCES documents(id) ON DELETE CASCADE" in SCHEMA_SQL
    assert "CREATE INDEX IF NOT EXISTS idx_chunks_document_id" in SCHEMA_SQL
    assert "idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)" in SCHEMA_SQL


def test_insert_chunks_rejects_incompatible_dimensions():
    """Verifies that insert_chunks raises ValueError when given vectors not matching 384 dimensions."""
    mock_conn = MagicMock()
    # 768-dimensional vector (Gemini) should be rejected
    gemini_vector = [0.01] * 768
    chunk_with_wrong_dim = [{
        "text": "Gemini text",
        "embedding": gemini_vector,
    }]

    with pytest.raises(ValueError) as exc_info:
        insert_chunks(mock_conn, document_id=1, chunks=chunk_with_wrong_dim)

    assert "Incompatible embedding dimension" in str(exc_info.value)
    assert "384" in str(exc_info.value)
    assert "768" in str(exc_info.value)


def test_insert_chunks_with_valid_384_dimensions():
    """Verifies that insert_chunks accepts valid 384-dimensional MiniLM embeddings."""
    mock_conn = MagicMock()
    minilm_vector = [0.05] * 384
    chunk_with_valid_dim = [{
        "text": "MiniLM chunk",
        "metadata": {"chunk_index": 0},
        "embedding": minilm_vector,
    }]

    count = insert_chunks(mock_conn, document_id=1, chunks=chunk_with_valid_dim)
    assert count == 1
    mock_conn.cursor.return_value.__enter__.return_value.executemany.assert_called_once()



def test_insert_document_and_chunks_mocked():
    """Verifies that insert_document and insert_chunks formulate SQL and params correctly."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = (42,)  # Mock generated document id

    doc_id = insert_document(
        mock_conn,
        filename="policy.md",
        document_type="markdown",
        content_hash="hash123",
    )
    assert doc_id == 42
    mock_cur.execute.assert_called_once()
    sql_arg, params_arg = mock_cur.execute.call_args[0]
    assert "INSERT INTO documents" in sql_arg
    assert params_arg == ("policy.md", "markdown", "hash123", "public", "public", "active")

    # Verify chunk insertion
    sample_chunks = [
        {
            "text": "Chunk text 1",
            "metadata": {"page_number": 1, "section": "Intro", "chunk_index": 0},
        },
        {
            "text": "Chunk text 2",
            "metadata": {"page_number": 1, "section": "Intro", "chunk_index": 1},
        },
    ]
    count = insert_chunks(mock_conn, document_id=42, chunks=sample_chunks)
    assert count == 2
    mock_cur.executemany.assert_called_once()
    chunk_sql, chunk_params = mock_cur.executemany.call_args[0]
    assert "INSERT INTO chunks" in chunk_sql
    assert len(chunk_params) == 2
    assert chunk_params[0][0] == 42  # document_id
    assert chunk_params[0][1] == "Chunk text 1"


# --- 2. Live PostgreSQL Integration Tests (Requires real PostgreSQL server) ---

IS_POSTGRES_AVAILABLE = check_db_connection()


@pytest.mark.skipif(
    not IS_POSTGRES_AVAILABLE,
    reason="Live PostgreSQL instance not detected on localhost:5432. Integration tests skipped.",
)
class TestLivePostgreSQLIntegration:
    """
    Live integration tests executed against a real PostgreSQL instance.
    Tests table creation, document/chunk insertion, foreign key enforcement, and cascade delete.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Initializes tables and cleans up test data after each test."""
        init_db()
        conn = get_connection(autocommit=True)
        yield conn
        # Cleanup test records
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE filename LIKE 'test_%';")
        conn.close()

    def test_live_db_table_creation(self, setup_and_teardown):
        conn = setup_and_teardown
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            tables = [row[0] for row in cur.fetchall()]
        assert "documents" in tables
        assert "chunks" in tables

    def test_live_db_insert_document_and_chunks(self, setup_and_teardown):
        conn = setup_and_teardown
        # 1. Insert document
        doc_id = insert_document(
            conn,
            filename="test_doc.md",
            document_type="markdown",
            content_hash="test_hash_001",
        )
        assert isinstance(doc_id, int)
        assert doc_id > 0

        # 2. Insert referencing chunks
        chunks = [
            {"text": "Live test chunk 1", "metadata": {"page_number": 1, "section": "Sec1", "chunk_index": 0}},
            {"text": "Live test chunk 2", "metadata": {"page_number": 1, "section": "Sec1", "chunk_index": 1}},
        ]
        inserted_count = insert_chunks(conn, document_id=doc_id, chunks=chunks)
        assert inserted_count == 2

        # 3. Query chunks back
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, chunk_index, page_number FROM chunks WHERE document_id = %s ORDER BY chunk_index ASC;",
                (doc_id,),
            )
            rows = cur.fetchall()

        assert len(rows) == 2
        assert rows[0][0] == "Live test chunk 1"
        assert rows[1][0] == "Live test chunk 2"

    def test_live_db_foreign_key_violation(self, setup_and_teardown):
        conn = setup_and_teardown
        invalid_doc_id = 999999999  # Non-existent document ID

        chunks = [{"text": "Orphan chunk", "metadata": {"page_number": 1, "section": "Err", "chunk_index": 0}}]

        # Foreign key must reject inserting chunk referencing non-existent document
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            insert_chunks(conn, document_id=invalid_doc_id, chunks=chunks)

    def test_live_db_cascade_delete(self, setup_and_teardown):
        conn = setup_and_teardown
        # Insert document and chunks
        doc_id = insert_document(conn, filename="test_cascade.txt", document_type="text")
        insert_chunks(conn, document_id=doc_id, chunks=[{"text": "Cascade child chunk", "metadata": {"chunk_index": 0}}])

        # Delete parent document
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))

        # Verify child chunks were automatically removed by ON DELETE CASCADE
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s;", (doc_id,))
            remaining_chunks = cur.fetchone()[0]

        assert remaining_chunks == 0

    def test_live_db_insert_and_query_vectors(self, setup_and_teardown):
        """Verifies storing 384-d vectors into PostgreSQL and querying them back."""
        conn = setup_and_teardown
        doc_id = insert_document(conn, filename="test_vectors.md", document_type="markdown")

        dummy_vector = [0.1] * 384
        chunks = [{
            "text": "Vector chunk content",
            "metadata": {"page_number": 1, "section": "Vectors", "chunk_index": 0},
            "embedding": dummy_vector,
        }]

        inserted_count = insert_chunks(conn, document_id=doc_id, chunks=chunks)
        assert inserted_count == 1

        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, embedding IS NOT NULL FROM chunks WHERE document_id = %s;",
                (doc_id,),
            )
            row = cur.fetchone()

        assert row[0] == "Vector chunk content"
        assert row[1] is True  # Embedding is present in pgvector column

    def test_live_db_rejects_incompatible_vector_dimension(self, setup_and_teardown):
        """Verifies that inserting a vector with incompatible dimensions (e.g. 768) is rejected with ValueError."""
        conn = setup_and_teardown
        doc_id = insert_document(conn, filename="test_invalid_dim.md", document_type="markdown")

        incompatible_vector = [0.1] * 768
        chunks = [{
            "text": "Incompatible vector chunk",
            "metadata": {"page_number": 1, "section": "Err", "chunk_index": 0},
            "embedding": incompatible_vector,
        }]

        with pytest.raises(ValueError) as exc_info:
            insert_chunks(conn, document_id=doc_id, chunks=chunks)

        assert "Incompatible embedding dimension" in str(exc_info.value)
        assert "384" in str(exc_info.value)

