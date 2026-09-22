"""
CLI Demonstration Tool for Vector Similarity Search with pgvector (Step 3).

Workflow:
1. Accepts user query text and optional top_k parameter via CLI arguments.
2. Embeds the query using the primary MiniLM model (384 dimensions).
3. Executes cosine vector similarity retrieval (<=>) against PostgreSQL.
4. Leverages the HNSW index on chunks.embedding for fast nearest-neighbor search.
5. Displays formatted matching chunks with scores, document origin, and metadata.

Usage:
    python scripts/search_vectors.py --query "What is chunking and chunk overlap?" --top-k 3
    python scripts/search_vectors.py
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    LOCAL_EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
from app.db import test_connection
from app.vector_store import retrieve_similar_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Search chunks in PostgreSQL using pgvector cosine similarity."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is the chunking strategy, chunk size, and chunk overlap?",
        help="Query text to search for (default: sample RAG question)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Number of most similar chunks to return (default: 3)",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("  Enterprise RAG - pgvector Cosine Vector Search Demo (Step 3)")
    print("=" * 75)
    print(f"  Target Database    : {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"  Embedding Model    : {LOCAL_EMBEDDING_MODEL}")
    print(f"  Vector Dimension   : {VECTOR_DIMENSION}")
    print(f"  Vector Index       : HNSW (cosine distance: <=>)")
    print(f"  Search Query       : \"{args.query}\"")
    print(f"  Top K Requested    : {args.top_k}")
    print("-" * 75)

    # 1. Connection check
    if not test_connection():
        print("[ERROR] Cannot connect to PostgreSQL database.")
        print("Please ensure your PostgreSQL container or service is running and configured.")
        print("Run: python scripts/init_db.py")
        sys.exit(1)

    # 2. Execute retrieval
    print("Generating query vector and executing vector search...")
    try:
        results = retrieve_similar_chunks(query_text=args.query, top_k=args.top_k)
    except Exception as e:
        print(f"\n[ERROR] Vector search failed: {e}")
        sys.exit(1)

    if not results:
        print("\n[RESULT] No matching chunks found.")
        print("Please verify that chunks have been stored in the database:")
        print("Run: python scripts/store_chunks_in_db.py")
        print("=" * 75)
        return

    print(f"\nRetrieved {len(results)} relevant chunk(s) (ordered by cosine similarity):\n")

    for rank, res in enumerate(results, start=1):
        meta = res.get("metadata", {})
        similarity = res.get("similarity", 0.0)
        distance = res.get("distance", 0.0)
        doc_name = meta.get("document_name", "unknown")
        section = meta.get("section") or "General"
        page = meta.get("page_number", 1)
        chunk_idx = meta.get("chunk_index", 0)
        content = res.get("content", "").strip()

        print(f"--- [Match #{rank}] Similarity: {similarity:.4f} | Cosine Distance: {distance:.4f} ---")
        print(f"  Document : {doc_name} (Page: {page}, Section: {section}, Chunk Index: {chunk_idx})")
        print(f"  Chunk ID : {res.get('chunk_id')}")
        print("  Content  :")
        # Print content with indentation
        for line in content.splitlines()[:8]:  # show up to first 8 lines
            print(f"    {line}")
        if len(content.splitlines()) > 8:
            print("    ...")
        print()

    print("=" * 75)
    print("Vector similarity search completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
