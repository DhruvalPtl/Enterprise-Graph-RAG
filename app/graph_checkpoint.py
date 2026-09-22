"""
Graph RAG Phase G2.2: SQLite-Backed Checkpoint and Progress Tracker.

Provides durable, thread-safe, and crash-resilient state tracking for
full-corpus knowledge graph extraction across 11,609 chunks.
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "graph_extraction_progress.db"


class GraphExtractionCheckpoint:
    """
    Manages progress and checkpoint state for the full extraction pipeline.
    Ensures that interrupted extraction runs resume seamlessly without re-processing completed chunks.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_CHECKPOINT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_tables()

    @contextmanager
    def _connection(self):
        """Context manager providing thread-safe, auto-closing SQLite connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_tables(self):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_progress (
                    chunk_id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER,
                    status TEXT NOT NULL, -- 'pending', 'processing', 'completed', 'failed', 'skipped_empty'
                    attempts INTEGER NOT NULL DEFAULT 0,
                    entity_count INTEGER DEFAULT 0,
                    relationship_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    model_used TEXT,
                    duration_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Add input_tokens and output_tokens columns if not existing
            for col in ("input_tokens", "output_tokens"):
                try:
                    conn.execute(f"ALTER TABLE chunk_progress ADD COLUMN {col} INTEGER DEFAULT 0;")
                except sqlite3.OperationalError:
                    pass

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_status ON chunk_progress(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_attempts ON chunk_progress(attempts);"
            )

    def register_chunks(self, chunks: List[Tuple[int, int, Optional[int]]]) -> int:
        """
        Idempotently registers chunks with status 'pending' if not already tracked.
        Returns the number of newly inserted chunks.
        chunks: List of (chunk_id, document_id, page_number)
        """
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR IGNORE INTO chunk_progress (chunk_id, document_id, page_number, status)
                VALUES (?, ?, ?, 'pending');
                """,
                chunks,
            )
            return cursor.rowcount

    def reset_processing_to_pending(self) -> int:
        """Resets any dangling 'processing' chunks back to 'pending' upon startup."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE chunk_progress
                SET status = 'pending', updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processing';
                """
            )
            return cursor.rowcount

    def reset_failed_to_pending(self, error_substring: Optional[str] = None) -> int:
        """Resets failed chunks back to pending with attempts reset to 0."""
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            if error_substring:
                cursor.execute(
                    """
                    UPDATE chunk_progress
                    SET status = 'pending', attempts = 0, error_message = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'failed' AND error_message LIKE ?;
                    """,
                    (f"%{error_substring}%",),
                )
            else:
                cursor.execute(
                    """
                    UPDATE chunk_progress
                    SET status = 'pending', attempts = 0, error_message = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'failed';
                    """
                )
            return cursor.rowcount

    def get_pending_chunks(
        self, batch_size: int = 25, max_attempts: int = 3
    ) -> List[Tuple[int, int, Optional[int]]]:
        """
        Retrieves next batch of chunks that are pending or failed with fewer than max_attempts.
        Returns List of (chunk_id, document_id, page_number).
        """
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, document_id, page_number
                FROM chunk_progress
                WHERE status = 'pending' OR (status = 'failed' AND attempts < ?)
                ORDER BY chunk_id ASC
                LIMIT ?;
                """,
                (max_attempts, batch_size),
            )
            return [(row["chunk_id"], row["document_id"], row["page_number"]) for row in cursor.fetchall()]

    def mark_processing(self, chunk_id: int):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE chunk_progress
                SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?;
                """,
                (chunk_id,),
            )

    def mark_completed(
        self,
        chunk_id: int,
        entity_count: int,
        relationship_count: int,
        model_used: Optional[str] = None,
        duration_ms: Optional[int] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE chunk_progress
                SET status = 'completed',
                    entity_count = ?,
                    relationship_count = ?,
                    model_used = ?,
                    duration_ms = ?,
                    input_tokens = ?,
                    output_tokens = ?,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?;
                """,
                (entity_count, relationship_count, model_used, duration_ms, input_tokens, output_tokens, chunk_id),
            )

    def mark_skipped_empty(self, chunk_id: int):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE chunk_progress
                SET status = 'skipped_empty',
                    entity_count = 0,
                    relationship_count = 0,
                    error_message = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?;
                """,
                (chunk_id,),
            )

    def mark_failed(self, chunk_id: int, error_message: str):
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                UPDATE chunk_progress
                SET status = 'failed',
                    attempts = attempts + 1,
                    error_message = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?;
                """,
                (error_message[:1000], chunk_id),
            )

    def get_progress_summary(self) -> Dict[str, Any]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    count(*) as total,
                    sum(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    sum(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    sum(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                    sum(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    sum(CASE WHEN status = 'skipped_empty' THEN 1 ELSE 0 END) as skipped_empty,
                    sum(CASE WHEN status = 'completed' THEN entity_count ELSE 0 END) as total_entities,
                    sum(CASE WHEN status = 'completed' THEN relationship_count ELSE 0 END) as total_relationships,
                    sum(CASE WHEN status = 'completed' THEN input_tokens ELSE 0 END) as total_input_tokens,
                    sum(CASE WHEN status = 'completed' THEN output_tokens ELSE 0 END) as total_output_tokens,
                    sum(attempts) as total_attempts
                FROM chunk_progress;
                """
            )
            row = cursor.fetchone()
            return {
                "total": row["total"] or 0,
                "completed": row["completed"] or 0,
                "pending": row["pending"] or 0,
                "processing": row["processing"] or 0,
                "failed": row["failed"] or 0,
                "skipped_empty": row["skipped_empty"] or 0,
                "total_entities": row["total_entities"] or 0,
                "total_relationships": row["total_relationships"] or 0,
                "total_input_tokens": row["total_input_tokens"] or 0,
                "total_output_tokens": row["total_output_tokens"] or 0,
                "total_attempts": row["total_attempts"] or 0,
            }

    def get_failed_chunks(self) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, document_id, page_number, attempts, error_message, updated_at
                FROM chunk_progress
                WHERE status = 'failed'
                ORDER BY chunk_id ASC;
                """
            )
            return [dict(row) for row in cursor.fetchall()]
