"""
Persists processed document chunks and their embeddings into PostgreSQL.

Workflow:
1. Verifies PostgreSQL connectivity.
2. Reads data/processed/embedded_chunks.json (or falls back to all_chunks.json).
3. Ensures tables, vector columns, and HNSW indexes are initialized.
4. Inserts source document records into the 'documents' table.
5. Inserts chunk records with 384-d embeddings into 'chunks' linked via document_id.
6. Preserves local JSON persistence.

Usage:
    python scripts/store_chunks_in_db.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ALL_CHUNKS_FILE, EMBEDDED_CHUNKS_FILE, VECTOR_DIMENSION
from app.db import test_connection, init_db, store_processed_chunks, get_db_config


def main():
    print("=" * 70)
    print("  Enterprise RAG - Store Vectors in PostgreSQL + pgvector")
    print("=" * 70)

    config = get_db_config()
    safe_host = config.get("host", "localhost")
    safe_port = config.get("port", 5432)
    safe_db = config.get("dbname", "rag_db")

    # Select best source file: prefer embedded_chunks.json with 384-d vectors
    source_file = EMBEDDED_CHUNKS_FILE if EMBEDDED_CHUNKS_FILE.exists() else ALL_CHUNKS_FILE

    print(f"  Source JSON File   : {source_file}")
    print(f"  Target Database    : {safe_host}:{safe_port}/{safe_db}")
    print(f"  Vector Dimension   : {VECTOR_DIMENSION} (MiniLM)")
    print("-" * 70)

    if not source_file.exists():
        print(f"[ERROR] Chunks file not found: {source_file}")
        print("Please run 'python run_pipeline.py' and 'python scripts/embed_chunks.py' first.")
        sys.exit(1)

    print("Checking PostgreSQL connection...")
    if not test_connection():
        print("\n[CONNECTION ERROR]")
        print(f"  Unable to connect to PostgreSQL at {safe_host}:{safe_port}/{safe_db}.")
        print("  Please start your PostgreSQL service or check settings in .env.")
        print("=" * 70)
        sys.exit(1)

    print("Ensuring database schema, pgvector extension, and HNSW index are ready...")
    init_db()

    print(f"Storing chunks and vectors from {source_file.name} into PostgreSQL...")
    try:
        stats = store_processed_chunks(source_file)
        print("\n[DATABASE STORAGE SUMMARY]")
        print(f"  Documents Inserted : {stats['documents_stored']}")
        print(f"  Chunks Inserted    : {stats['chunks_stored']}")
        print(f"  Vector Dimensions  : {VECTOR_DIMENSION} (MiniLM vectors stored in pgvector)")
        print("  Referential Link   : chunks.document_id -> documents.id [ON DELETE CASCADE]")
        print("  Index Available    : HNSW cosine index ('idx_chunks_embedding_hnsw')")
        print(f"  JSON Persistence   : Intact ({source_file} preserved)")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Failed to persist chunks to PostgreSQL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

