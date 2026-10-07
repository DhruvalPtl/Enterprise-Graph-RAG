"""
Database Reset & Clean Slate Utility for Enterprise Graph RAG.

Safely truncates all documents, chunks, entities, and relationships in PostgreSQL,
allowing the user to start completely fresh with their own documents out of the box.

Usage:
    python scripts/reset_db.py
    python scripts/reset_db.py --yes
    python scripts/reset_db.py --include-files  # Also removes data/processed/ cache
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_connection, test_connection, init_db, get_db_config
from app.config import ALL_CHUNKS_FILE, EMBEDDED_CHUNKS_FILE


def reset_database(clear_processed_files: bool = False, force: bool = False) -> None:
    """Safely resets the PostgreSQL database to a clean, empty state."""
    config = get_db_config()
    safe_host = config.get("host", "localhost")
    safe_port = config.get("port", 5432)
    safe_db = config.get("dbname", "rag_db")

    print("=" * 70)
    print("  Enterprise Graph RAG - Database Reset Utility")
    print("=" * 70)
    print(f"  Target Database : {safe_host}:{safe_port}/{safe_db}")
    print(f"  Clear Local JSON: {clear_processed_files}")
    print("-" * 70)

    if not test_connection():
        print(f"[ERROR] Unable to connect to PostgreSQL at {safe_host}:{safe_port}/{safe_db}.")
        print("Please ensure PostgreSQL is running.")
        sys.exit(1)

    if not force:
        print("\nWARNING: This will permanently delete ALL stored documents, chunks,")
        print("entities, and relationships from the PostgreSQL database.")
        confirmation = input("Type 'RESET' to confirm database wipe: ").strip()
        if confirmation != "RESET":
            print("\nReset aborted by user. No data was modified.")
            sys.exit(0)

    print("\nTruncating tables...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Cascading truncate on core tables
            cur.execute("""
                TRUNCATE TABLE documents CASCADE;
            """)
            # Check if entities table exists and truncate
            cur.execute("""
                DO $$
                BEGIN
                    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'entities') THEN
                        TRUNCATE TABLE entities CASCADE;
                    END IF;
                END $$;
            """)
        conn.commit()

    print("[SUCCESS] All database records truncated.")

    # Re-verify schema idempotently
    print("Verifying schema & HNSW indexes...")
    init_db()

    if clear_processed_files:
        if ALL_CHUNKS_FILE.exists():
            ALL_CHUNKS_FILE.unlink()
            print(f"  Deleted: {ALL_CHUNKS_FILE}")
        if EMBEDDED_CHUNKS_FILE.exists():
            EMBEDDED_CHUNKS_FILE.unlink()
            print(f"  Deleted: {EMBEDDED_CHUNKS_FILE}")

    print("\nDatabase is now completely fresh and ready for your own custom documents!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Reset Enterprise Graph RAG PostgreSQL database.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument(
        "--include-files",
        action="store_true",
        help="Also remove data/processed/all_chunks.json and embedded_chunks.json",
    )
    args = parser.parse_args()

    reset_database(clear_processed_files=args.include_files, force=args.yes)


if __name__ == "__main__":
    main()
