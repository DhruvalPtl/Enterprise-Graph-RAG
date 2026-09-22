"""
Initializes the PostgreSQL database schema for Enterprise RAG (Step 3).

Creates:
- pgvector extension (CREATE EXTENSION IF NOT EXISTS vector)
- documents table
- chunks table (with embedding VECTOR(384) and foreign key ON DELETE CASCADE)
- idx_chunks_document_id B-tree index
- idx_chunks_embedding_hnsw HNSW cosine vector index (vector_cosine_ops)

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db, test_connection, get_db_config
from app.config import VECTOR_DIMENSION


def main():
    print("=" * 70)
    print("  Enterprise RAG - PostgreSQL + pgvector Initialization (Step 3)")
    print("=" * 70)

    config = get_db_config()
    safe_host = config.get("host", "localhost")
    safe_port = config.get("port", 5432)
    safe_db = config.get("dbname", "rag_db")
    safe_user = config.get("user", "postgres")

    print(f"  Target Host        : {safe_host}:{safe_port}")
    print(f"  Database Name      : {safe_db}")
    print(f"  Database User      : {safe_user}")
    print(f"  Vector Dimension   : {VECTOR_DIMENSION} (MiniLM)")
    print("-" * 70)

    print("Checking connection to PostgreSQL...")
    if not test_connection():
        print("\n[CONNECTION ERROR]")
        print(f"  Unable to connect to PostgreSQL at {safe_host}:{safe_port}/{safe_db}.")
        print("  Please ensure:")
        print("    1. PostgreSQL server / Docker container is running.")
        print("    2. The database exists (e.g., 'rag_db').")
        print("    3. Credentials in .env (POSTGRES_USER, POSTGRES_PASSWORD) are correct.")
        print("=" * 70)
        sys.exit(1)

    print("Connection successful! Initializing extension, tables, and indexes...")
    try:
        init_db()
        print("\n[SUCCESS]")
        print("  Extension     : vector (pgvector)")
        print("  Tables        : 'documents', 'chunks', 'entities', 'relationships'")
        print(f"  Vector Column : chunks.embedding VECTOR({VECTOR_DIMENSION})")
        print("  Foreign Keys  : chunks.document_id -> documents(id) [ON DELETE CASCADE]")
        print("                  relationships.source_entity_id -> entities(id) [ON DELETE CASCADE]")
        print("                  relationships.target_entity_id -> entities(id) [ON DELETE CASCADE]")
        print("                  relationships.document_id -> documents(id) [ON DELETE CASCADE]")
        print("                  relationships.chunk_id -> chunks(id) [ON DELETE CASCADE]")
        print("  B-Tree Indexes: 'idx_chunks_document_id', 'idx_entities_canonical_name', 'idx_entities_type',")
        print("                  'idx_relationships_source', 'idx_relationships_target', 'idx_relationships_type',")
        print("                  'idx_relationships_document_id', 'idx_relationships_chunk_id'")
        print("  HNSW Index    : 'idx_chunks_embedding_hnsw' on chunks(embedding vector_cosine_ops)")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Failed to execute schema DDL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

