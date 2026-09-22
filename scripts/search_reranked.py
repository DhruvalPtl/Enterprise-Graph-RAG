"""
CLI Demonstration Tool for Cross-Encoder Reranking (Step 5).

Demonstrates:
1. Stage 1: Fast hybrid candidate retrieval (Dense Vector + Sparse BM25 fused with RRF).
2. Stage 2: Deep cross-attention reranking using cross-encoder/ms-marco-MiniLM-L-6-v2.
3. Comparative analysis showing rank migrations, score shifts, and final top results.

Usage:
    python scripts/search_reranked.py --query "What is chunking and chunk overlap strategy?" --candidate-k 10 --top-k 5
    python scripts/search_reranked.py
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    LOCAL_EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    RERANKER_MODEL,
    RERANKER_CANDIDATE_K,
    RERANKER_TOP_K,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
from app.db import test_connection
from app.hybrid import HybridRetriever
from app.reranker import CrossEncoderReranker, RerankedRetrievalPipeline


def format_excerpt(content: str, max_lines: int = 4) -> str:
    """Formats content with indentation for clean terminal display."""
    lines = content.strip().splitlines()
    excerpt = "\n".join(f"    {line}" for line in lines[:max_lines])
    if len(lines) > max_lines:
        excerpt += "\n    ..."
    return excerpt


def main():
    parser = argparse.ArgumentParser(
        description="Search chunks using Two-Stage Retrieval (Hybrid RRF -> Cross-Encoder Reranker)."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is chunking and chunk overlap strategy?",
        help="Query text to search for (default: sample RAG question)",
    )
    parser.add_argument(
        "--candidate-k",
        "-c",
        type=int,
        default=RERANKER_CANDIDATE_K,
        help=f"Number of Stage 1 hybrid candidates to retrieve for reranking (default: {RERANKER_CANDIDATE_K})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=RERANKER_TOP_K,
        help=f"Number of final Stage 2 reranked results to return (default: {RERANKER_TOP_K})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=RERANKER_MODEL,
        help=f"Cross-encoder model name (default: {RERANKER_MODEL})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  Enterprise RAG - Cross-Encoder Reranking (Step 5)")
    print("=" * 80)
    print(f"  Target Database      : {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"  Bi-Encoder (Stage 1) : {LOCAL_EMBEDDING_MODEL} ({VECTOR_DIMENSION}-d)")
    print(f"  Cross-Encoder (St. 2): {args.model}")
    print(f"  Query                : \"{args.query}\"")
    print(f"  Pipeline Stages      : Hybrid RRF ({args.candidate_k} candidates) -> Cross-Encoder ({args.top_k} final)")
    print("-" * 80)

    # 1. Database connection check
    if not test_connection():
        print("[ERROR] Cannot connect to PostgreSQL database.")
        print("Please ensure your PostgreSQL container is running: docker start enterprise_rag_postgres")
        sys.exit(1)

    # 2. Run two-stage pipeline with full diagnostics
    print("Executing Stage 1 (Vector + BM25 -> RRF) and Stage 2 (Cross-Encoder)...")
    hybrid_retriever = HybridRetriever(
        vector_top_k=args.candidate_k,
        bm25_top_k=args.candidate_k,
        rrf_top_k=args.candidate_k,
    )
    reranker = CrossEncoderReranker(model_name=args.model)
    pipeline = RerankedRetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )

    diagnostics = pipeline.retrieve_with_diagnostics(
        query_text=args.query,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )

    candidates = diagnostics["candidates"]
    reranked_results = diagnostics["reranked_results"]

    # Build mapping for Stage 1 initial ranks
    candidate_rank_map = {
        cand["chunk_id"]: cand.get("rank", idx + 1)
        for idx, cand in enumerate(candidates)
    }

    # --- 1. STAGE 1 CANDIDATES ---
    print("\n" + "+" * 80)
    print(f"  STAGE 1: HYBRID CANDIDATE POOL (Top {len(candidates)} from RRF)")
    print("+" * 80)
    for c in candidates[:min(10, len(candidates))]:
        doc = c.get("metadata", {}).get("document_name", "unknown")
        section = c.get("metadata", {}).get("section", "General")
        sources = ", ".join(f"{s}:{info['rank']}" for s, info in c.get("sources", {}).items())
        print(f"  RRF Rank #{c['rank']} | Score: {c['rrf_score']:.6f} | Chunk ID: {c['chunk_id']} | Sources: [{sources}] | Doc: {doc} ({section})")

    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more candidates in Stage 1 pool.")

    # --- 2. STAGE 2 RERANKED RESULTS ---
    print("\n" + "=" * 80)
    print(f"  STAGE 2: FINAL CROSS-ENCODER RERANKED RESULTS (Top {len(reranked_results)})")
    print(f"  Model: {args.model}")
    print("=" * 80)

    for res in reranked_results:
        orig_rank = candidate_rank_map.get(res.chunk_id, "?")
        if isinstance(orig_rank, int):
            delta = orig_rank - res.rank
            delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "=")
        else:
            delta_str = "NEW"

        doc = res.metadata.get("document_name", "unknown")
        page = res.metadata.get("page_number", 1)
        section = res.metadata.get("section", "General")

        print(f"\n--- [Final Rank #{res.rank}] (Stage 1 RRF Rank: #{orig_rank}, Movement: {delta_str}) ---")
        print(f"  Cross-Encoder Score : {res.reranker_score:.6f}")
        print(f"  Stage 1 RRF Score   : {res.rrf_score:.6f}" if res.rrf_score is not None else "  Stage 1 RRF Score   : N/A")
        print(f"  Chunk ID            : {res.chunk_id}")
        print(f"  Document / Section  : {doc} (Page {page}, Section: {section})")
        print("  Content Excerpt     :")
        print(format_excerpt(res.content, max_lines=5))

    # --- 3. RANK MIGRATION SUMMARY TABLE ---
    print("\n" + "-" * 80)
    print("  RANK MIGRATION SUMMARY (Impact of Cross-Encoder Reranking)")
    print("-" * 80)
    print(f"  {'Final Rank':<12} {'Initial RRF':<14} {'Movement':<10} {'CE Score':<12} {'RRF Score':<12} {'Chunk ID':<10} {'Document':<18}")
    print(f"  {'-'*10:<12} {'-'*12:<14} {'-'*8:<10} {'-'*10:<12} {'-'*10:<12} {'-'*8:<10} {'-'*16:<18}")

    for res in reranked_results:
        orig_rank = candidate_rank_map.get(res.chunk_id, "?")
        if isinstance(orig_rank, int):
            delta = orig_rank - res.rank
            delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "0")
        else:
            delta_str = "NEW"
        doc = res.metadata.get("document_name", "unknown")[:16]
        rrf_str = f"{res.rrf_score:.4f}" if res.rrf_score is not None else "N/A"
        print(f"  #{res.rank:<11} #{orig_rank:<13} {delta_str:<10} {res.reranker_score:<12.4f} {rrf_str:<12} {res.chunk_id:<10} {doc:<18}")

    print("=" * 80)
    print("Cross-Encoder reranking demonstration completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
