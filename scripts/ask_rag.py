"""
CLI Demonstration Tool for End-to-End Enterprise RAG (Step 6).

Demonstrates:
1. Two-stage retrieval: Dense vector + BM25 -> RRF -> Cross-Encoder reranker.
2. Context preparation and authoritative citation generation.
3. Grounded generation via Google Gemini LLM.
4. Clean presentation of answer and cited sources.

Usage:
    python scripts/ask_rag.py --query "What is chunking and chunk overlap strategy?"
    python scripts/ask_rag.py --query "What are the data privacy requirements for enterprise AI?"
    python scripts/ask_rag.py --query "What is the capital of Mars?"
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
    RERANKER_CANDIDATE_K,
    RERANKER_TOP_K,
    MAX_CONTEXT_CHARS,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
)
from app.db import test_connection
from app.rag import RAGPipeline
from app.reranker import RerankedRetrievalPipeline
from app.context_builder import ContextBuilder
from app.prompts import build_rag_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions to the Enterprise RAG Knowledge Intelligence Platform."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What is chunking and chunk overlap strategy?",
        help="Question to ask the RAG platform",
    )
    parser.add_argument(
        "--candidate-k",
        "-c",
        type=int,
        default=RERANKER_CANDIDATE_K,
        help=f"Number of Stage 1 hybrid candidates (default: {RERANKER_CANDIDATE_K})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=RERANKER_TOP_K,
        help=f"Number of Stage 2 reranked evidence passages (default: {RERANKER_TOP_K})",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help=f"Maximum context characters for the prompt (default: {MAX_CONTEXT_CHARS})",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  ENTERPRISE RAG - KNOWLEDGE INTELLIGENCE PLATFORM (Step 6)")
    print("=" * 80)
    print(f"  Target Database : {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"  LLM Model       : {GEMINI_LLM_MODEL}")
    print(f"  Question        : \"{args.query}\"")
    print("-" * 80)

    # 1. Database connectivity check
    if not test_connection():
        print("[ERROR] Cannot connect to PostgreSQL database.")
        print("Please ensure your PostgreSQL container is running: docker start enterprise_rag_postgres")
        sys.exit(1)

    api_key_available = bool(GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY"))

    # 2. Execute RAG pipeline
    if api_key_available:
        try:
            pipeline = RAGPipeline(
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                max_context_chars=args.max_chars,
            )
            print("Retrieving evidence, reranking, and generating grounded answer...\n")
            response = pipeline.answer_query(query=args.query)

            print("=" * 80)
            print("  RETRIEVAL")
            print("=" * 80)
            print(f"  Stage 1 Candidates Retrived : {response.diagnostics.get('candidate_count', 'N/A')}")
            print(f"  Stage 2 Evidence Passages   : {response.diagnostics.get('evidence_count', 'N/A')}")
            print(f"  Context Size                : {response.diagnostics.get('context_chars', 'N/A')} characters")

            print("\n" + "=" * 80)
            print("  GENERATED ANSWER")
            print("=" * 80)
            print(response.answer)

            print("\n" + "=" * 80)
            print("  AUTHORITATIVE SOURCES")
            print("=" * 80)
            if response.citations:
                for c in response.citations:
                    score_info = f"(Reranker: {c.reranker_score:.4f})" if c.reranker_score is not None else ""
                    print(f"  {c.format_citation()} {score_info}")
            else:
                print("  (No sources cited)")

        except Exception as exc:
            print(f"[ERROR] Generation failed: {exc}")
            sys.exit(1)
    else:
        # Dry-run execution through retrieval, reranking, context building, and citations
        print("[NOTICE] GEMINI_API_KEY is not configured in environment or .env.")
        print("Executing Stage 1 & 2 retrieval and context construction (dry run up to LLM boundary)...\n")

        retrieval_pipeline = RerankedRetrievalPipeline(
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
        diagnostics = retrieval_pipeline.retrieve_with_diagnostics(
            query_text=args.query,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
        candidates = diagnostics["candidates"]
        reranked = diagnostics["reranked_results"]

        context_builder = ContextBuilder(max_context_chars=args.max_chars)
        built_context = context_builder.build_context(
            query=args.query,
            results=reranked,
        )

        print("=" * 80)
        print("  RETRIEVAL & EVIDENCE SUMMARY")
        print("=" * 80)
        print(f"  Stage 1 Hybrid Candidates   : {len(candidates)}")
        print(f"  Stage 2 Reranked Evidence   : {len(reranked)}")
        print(f"  Context Budget Used         : {built_context.total_chars} / {args.max_chars} chars")

        print("\n" + "=" * 80)
        print("  AUTHORITATIVE SOURCES (Derived from SearchResult metadata)")
        print("=" * 80)
        for c in built_context.citations:
            ce_score = f"(Cross-Encoder: {c.reranker_score:.4f})" if c.reranker_score is not None else ""
            print(f"  {c.format_citation()} {ce_score}")

        print("\n" + "=" * 80)
        print("  STRUCTURED CONTEXT SENT TO LLM (Preview)")
        print("=" * 80)
        lines = built_context.context_text.splitlines()
        preview = "\n".join(f"  {line}" for line in lines[:20])
        print(preview)
        if len(lines) > 20:
            print(f"  ... and {len(lines) - 20} more lines.")

        print("\n" + "-" * 80)
        print("  To enable live LLM answer generation:")
        print("  1. Get an API key from https://aistudio.google.com/app/apikey")
        print("  2. Set GEMINI_API_KEY=your_key in .env or run: $env:GEMINI_API_KEY='your_key'")
        print("  3. Re-run: python scripts/ask_rag.py -q \"...\"")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Enterprise RAG demonstration completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
