"""
Unit tests for Graph RAG Phase G2.2 Checkpoint Management and PostgreSQL Transaction Safety.
"""
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.graph_checkpoint import GraphExtractionCheckpoint
from app.graph_ontology import EntityExtraction, RelationshipExtraction, ValidatedExtractionResult
from app.db import get_connection, insert_chunk_graph_transaction


class TestGraphExtractionCheckpoint:
    """Tests SQLite-backed checkpoint state tracking and recovery."""

    @pytest.fixture
    def checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_progress.db"
            yield GraphExtractionCheckpoint(db_path=db_path)

    def test_registration_and_pending_retrieval(self, checkpoint):
        chunks = [(101, 1, 1), (102, 1, 1), (103, 1, 2), (104, 2, 1)]
        new_count = checkpoint.register_chunks(chunks)
        assert new_count == 4

        # Re-registration should be idempotent (0 inserted)
        dup_count = checkpoint.register_chunks(chunks)
        assert dup_count == 0

        pending = checkpoint.get_pending_chunks(batch_size=2)
        assert len(pending) == 2
        assert pending[0] == (101, 1, 1)
        assert pending[1] == (102, 1, 1)

    def test_state_transitions(self, checkpoint):
        chunks = [(1, 1, 1), (2, 1, 1), (3, 1, 2)]
        checkpoint.register_chunks(chunks)

        checkpoint.mark_processing(1)
        checkpoint.mark_completed(1, entity_count=3, relationship_count=2, model_used="test-model", duration_ms=500)

        checkpoint.mark_failed(2, error_message="API Rate Limit 429")
        checkpoint.mark_skipped_empty(3)

        summary = checkpoint.get_progress_summary()
        assert summary["total"] == 3
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert summary["skipped_empty"] == 1
        assert summary["pending"] == 0
        assert summary["total_entities"] == 3
        assert summary["total_relationships"] == 2

    def test_retry_limits(self, checkpoint):
        checkpoint.register_chunks([(10, 1, 1)])

        checkpoint.mark_failed(10, "Attempt 1 failed")
        checkpoint.mark_failed(10, "Attempt 2 failed")

        # With max_attempts=3, should still be retrieved
        retrievable = checkpoint.get_pending_chunks(batch_size=10, max_attempts=3)
        assert len(retrievable) == 1
        assert retrievable[0][0] == 10

        # Mark 3rd failure
        checkpoint.mark_failed(10, "Attempt 3 failed")
        failed_chunks = checkpoint.get_failed_chunks()
        assert len(failed_chunks) == 1
        assert failed_chunks[0]["attempts"] == 3

        # Exceeds max_attempts=3, should no longer be pending
        none_pending = checkpoint.get_pending_chunks(batch_size=10, max_attempts=3)
        assert len(none_pending) == 0

    def test_reset_processing_to_pending(self, checkpoint):
        checkpoint.register_chunks([(1, 1, 1), (2, 1, 1)])
        checkpoint.mark_processing(1)

        summary = checkpoint.get_progress_summary()
        assert summary["processing"] == 1
        assert summary["pending"] == 1

        reset_count = checkpoint.reset_processing_to_pending()
        assert reset_count == 1

        summary2 = checkpoint.get_progress_summary()
        assert summary2["processing"] == 0
        assert summary2["pending"] == 2


class TestPostgresTransactionSafety:
    """Tests transactional atomicity and rollback safety during graph ingestion."""

    def test_successful_transactional_insertion(self):
        conn = get_connection()
        try:
            # Look up an existing document and chunk from DB
            with conn.cursor() as cur:
                cur.execute("SELECT id, document_id, page_number FROM chunks LIMIT 1;")
                row = cur.fetchone()
                assert row is not None, "PostgreSQL chunks table is empty."
                chunk_id, doc_id, page_num = row

            result = ValidatedExtractionResult(
                document_id=doc_id,
                chunk_id=chunk_id,
                page_number=page_num,
                entities=[
                    EntityExtraction(name="Test Entity Alpha", entity_type="TECHNOLOGY"),
                    EntityExtraction(name="Test Entity Beta", entity_type="CONCEPT"),
                ],
                relationships=[
                    RelationshipExtraction(
                        source="Test Entity Alpha",
                        target="Test Entity Beta",
                        relationship_type="USES",
                        evidence="Alpha uses Beta.",
                    )
                ],
            )

            # Insert in transaction
            ent_cnt, rel_cnt = insert_chunk_graph_transaction(conn, result)
            assert ent_cnt == 2
            assert rel_cnt == 1

            # Verify in DB
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM entities WHERE canonical_name IN ('Test Entity Alpha', 'Test Entity Beta');"
                )
                assert len(cur.fetchall()) == 2

            # Clean up test records
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM entities WHERE canonical_name IN ('Test Entity Alpha', 'Test Entity Beta');"
                    )
        finally:
            conn.close()

    def test_transaction_rollback_on_foreign_key_failure(self):
        conn = get_connection()
        try:
            # Use non-existent chunk_id (999999999) to force foreign key violation
            fake_chunk_id = 999999999
            fake_doc_id = 999999999

            result = ValidatedExtractionResult(
                document_id=fake_doc_id,
                chunk_id=fake_chunk_id,
                page_number=1,
                entities=[
                    EntityExtraction(name="Rollback Entity Gamma", entity_type="TECHNOLOGY"),
                ],
                relationships=[
                    RelationshipExtraction(
                        source="Rollback Entity Gamma",
                        target="Rollback Entity Gamma",
                        relationship_type="USES",
                    )
                ],
            )
            # Relationship with non-existent foreign keys should cause rollback
            # First clean any leftover
            with conn.cursor() as cur:
                cur.execute("DELETE FROM entities WHERE canonical_name = 'Rollback Entity Gamma';")
            conn.commit()

            # Attempt insertion with invalid FK -> should raise Exception and roll back
            with pytest.raises(Exception):
                insert_chunk_graph_transaction(conn, result)

            # Entity should NOT have been committed
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM entities WHERE canonical_name = 'Rollback Entity Gamma';")
                assert cur.fetchone() is None
        finally:
            conn.close()
