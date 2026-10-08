"""
CLI Demonstration Tool for Hybrid Retrieval and Reciprocal Rank Fusion.

Demonstrates:
1. Semantic dense vector retrieval (PostgreSQL + pgvector HNSW).
2. Lexical sparse keyword retrieval (Okapi BM25).
3. Side-by-side comparison of individual rankings.
4. Reciprocal Rank Fusion (RRF) combining vector and BM25 rankings into a unified list.

Usage:
    python scripts/search_hybrid.py --query "What is chunking and chunk overlap strategy?" --top-k 5
    python scripts/search_hybrid.py
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    LOCAL_EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    VECTOR_TOP_K,
    BM25_TOP_K,
    RRF_TOP_K,
    RRF_K,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
from app.db import test_connection
from app.hybrid import HybridRetriever


def format_excerpt(content: str, max_lines: int = 4) -> str:
    """Formats content with indentation for clean terminal display."""
    lines = content.strip().splitlines()
    excerpt = "\n".join(f"    {line}" for line in lines[:max_lines])
    if len(lines) > max_lines:
        excerpt += "\n    ..."
    return excerpt


def main():
    parser = argparse.ArgumentParser(
        description="Search chunks using Hybrid Retrieval (Vector + BM25 + RRF)."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is chunking and chunk overlap strategy?",
        help="Query text to search for (default: sample RAG question)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help=f"Number of final fused results to return (default: 5)",
    )
    parser.add_argument(
        "--vector-k",
        type=int,
        default=5,
        help="Number of vector candidates to retrieve (default: 5)",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=5,
        help="Number of BM25 candidates to retrieve (default: 5)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help=f"RRF smoothing constant (default: {RRF_K})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  Enterprise RAG - Hybrid Retrieval & Reciprocal Rank Fusion")
    print("=" * 80)
    print(f"  Target Database    : {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"  Vector Model       : {LOCAL_EMBEDDING_MODEL} ({VECTOR_DIMENSION}-d)")
    print(f"  Lexical Retriever  : Okapi BM25 (Lucene non-negative IDF)")
    print(f"  Fusion Algorithm   : Reciprocal Rank Fusion (k={args.rrf_k})")
    print(f"  Query              : \"{args.query}\"")
    print(f"  Candidate Limits   : Vector={args.vector_k}, BM25={args.bm25_k} -> Fused Top-K={args.top_k}")
    print("-" * 80)

    # 1. Connection check
    if not test_connection():
        print("[ERROR] Cannot connect to PostgreSQL database.")
        print("Please ensure your PostgreSQL container is running: docker start enterprise_rag_postgres")
        sys.exit(1)

    # 2. Execute hybrid retrieval with full system breakdown
    print("Executing independent Vector search and BM25 search, then applying RRF...\n")
    retriever = HybridRetriever(
        rrf_k=args.rrf_k,
        vector_top_k=args.vector_k,
        bm25_top_k=args.bm25_k,
        rrf_top_k=args.top_k,
    )

    details = retriever.retrieve_with_details(
        query_text=args.query,
        top_k=args.top_k,
        vector_top_k=args.vector_k,
        bm25_top_k=args.bm25_k,
        rrf_k=args.rrf_k,
    )

    vector_results = details["vector_results"]
    bm25_results = details["bm25_results"]
    fused_results = details["fused_results"]

    # --- 1. VECTOR RESULTS ---
    print("+" * 80)
    print(f"  [1] DENSE VECTOR RESULTS (Top {len(vector_results)})")
    print("+" * 80)
    if not vector_results:
        print("  (No vector results found)\n")
    for r in vector_results:
        doc = r.get("metadata", {}).get("document_name", "unknown")
        sim = r.get("similarity", 0.0)
        dist = r.get("distance", 0.0)
        print(f"  Rank #{r['rank']} | Chunk ID: {r['chunk_id']} | Sim: {sim:.4f} (Dist: {dist:.4f}) | Doc: {doc}")
        print(format_excerpt(r.get("content", "")))
        print()

    # --- 2. BM25 RESULTS ---
    print("+" * 80)
    print(f"  [2] SPARSE BM25 RESULTS (Top {len(bm25_results)})")
    print("+" * 80)
    if not bm25_results:
        print("  (No BM25 results found)\n")
    for r in bm25_results:
        doc = r.get("metadata", {}).get("document_name", "unknown")
        score = r.get("score", 0.0)
        print(f"  Rank #{r['rank']} | Chunk ID: {r['chunk_id']} | BM25 Score: {score:.4f} | Doc: {doc}")
        print(format_excerpt(r.get("content", "")))
        print()

    # --- 3. FUSED HYBRID / RRF RESULTS ---
    print("=" * 80)
    print(f"  [3] FUSED HYBRID RESULTS via Reciprocal Rank Fusion (Top {len(fused_results)})")
    print(f"      Formula: Score(d) = SUM 1 / ({args.rrf_k} + rank_m(d))")
    print("=" * 80)

    if not fused_results:
        print("  (No fused results returned)\n")
    for r in fused_results:
        doc = r.get("metadata", {}).get("document_name", "unknown")
        page = r.get("metadata", {}).get("page_number", 1)
        section = r.get("metadata", {}).get("section", "General")
        sources_info = []
        for src, info in r.get("sources", {}).items():
            sources_info.append(f"{src.upper()}: rank #{info['rank']}")
        sources_str = ", ".join(sources_info)

        print(f"--- [Match #{r['rank']}] Fused RRF Score: {r['rrf_score']:.6f} ---")
        print(f"  Chunk ID  : {r['chunk_id']} (Document: {doc}, Page: {page}, Section: {section})")
        print(f"  Sources   : {sources_str}")
        print("  Content   :")
        print(format_excerpt(r.get("content", ""), max_lines=6))
        print()

    print("=" * 80)
    print("Hybrid retrieval demonstration completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
