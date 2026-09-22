"""
Graph RAG Phase G5: Empirical Evaluation and Benchmark Runner CLI.

Executes side-by-side benchmarking between:
  System A: Vector-Only RAG (Vector + BM25 + RRF + Cross-Encoder + Context + Gemini)
  System B: Hybrid Graph + Vector RAG (Vector + BM25 + RRF + GraphRetriever + Fusion + Cross-Encoder + Context + Gemini)

Usage:
    python scripts/evaluate_graph_rag.py [--limit 5] [--category multi_hop_relationship] [--no-llm]
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection
from app.evaluation.dataset import load_evaluation_dataset, DEFAULT_EVALUATION_DATASET_PATH
from app.evaluation.reporting import EvaluationReporter
from app.evaluation.runner import EvaluationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_graph_rag")


def main():
    parser = argparse.ArgumentParser(description="Graph RAG Phase G5 Evaluation & Benchmark Runner")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_EVALUATION_DATASET_PATH), help="Path to evaluation JSON dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation questions to run")
    parser.add_argument("--category", type=str, default=None, help="Filter to specific question category")
    parser.add_argument("--candidate-k", type=int, default=10, help="Number of retrieval candidates per path (default: 10)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top reranked chunks for context (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature (default: 0.0)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM generation and evaluate pure retrieval metrics (fast mode)")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "data" / "evaluation" / "results"), help="Directory to save evaluation results")
    args = parser.parse_args()

    print("=" * 80)
    print("  Graph RAG Phase G5: Empirical Evaluation & Benchmarking")
    print("=" * 80)
    print(f"  Dataset Path        : {args.dataset}")
    print(f"  Candidate K         : {args.candidate_k}")
    print(f"  Reranker Top K      : {args.top_k}")
    print(f"  LLM Generation      : {'DISABLED (Pure Retrieval Mode)' if args.no_llm else 'ENABLED (End-to-End Mode)'}")
    print(f"  Filter Category     : {args.category or 'ALL (8 Categories)'}")
    if args.limit:
        print(f"  Question Limit      : {args.limit}")
    print("-" * 80)

    # 1. Load Evaluation Dataset
    dataset = load_evaluation_dataset(args.dataset)
    print(f"Loaded {len(dataset)} evaluation questions across categories: {sorted(list(dataset.categories))}\n")

    # 2. Initialize Runner
    runner = EvaluationRunner(
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        temperature=args.temperature,
        enable_llm=not args.no_llm,
    )

    # 3. Execute Benchmark
    start_time = time.time()
    conn = get_connection()
    try:
        benchmark_payload = runner.run_benchmark(
            dataset=dataset,
            limit=args.limit,
            category=args.category,
            conn=conn,
        )
    finally:
        conn.close()

    elapsed = time.time() - start_time
    summary = benchmark_payload["summary"]

    # 4. Save Artifacts
    output_dir = Path(args.output_dir)
    results_json_path = output_dir / "graph_rag_results.json"
    summary_json_path = output_dir / "graph_rag_summary.json"
    report_md_path = output_dir / "graph_rag_report.md"

    # Also save to main reports directory
    main_reports_dir = BASE_DIR / "reports"
    main_report_md_path = main_reports_dir / "graph_rag_evaluation_report.md"

    reporter = EvaluationReporter()
    reporter.save_json_results(benchmark_payload, results_json_path)
    reporter.save_json_summary(summary, summary_json_path)

    md_report = reporter.generate_markdown_report(benchmark_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    main_reports_dir.mkdir(parents=True, exist_ok=True)
    with open(main_report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n[EXPORT] Saved full trace JSON -> {results_json_path}")
    print(f"[EXPORT] Saved summary JSON    -> {summary_json_path}")
    print(f"[EXPORT] Saved Markdown Report -> {report_md_path}")
    print(f"[EXPORT] Saved Project Report  -> {main_report_md_path}\n")

    # 5. Print Console Summary Table
    ret = summary.get("retrieval", {})
    vec = ret.get("vector_only", {})
    hyb = ret.get("hybrid", {})
    lat = summary.get("latency", {})
    verdicts = summary.get("overall_verdicts", {})
    graph_s = summary.get("graph_specific", {})

    print("=" * 80)
    print("  EMPIRICAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Completed in: {elapsed:.2f}s | Questions: {benchmark_payload['metadata']['total_questions']}")
    print(f"  Verdicts    : Hybrid Won: {verdicts.get('hybrid_won', 0)} | Vector Won: {verdicts.get('vector_won', 0)} | Tie: {verdicts.get('tie', 0)}")
    print("-" * 80)
    print(f"  {'Metric':<25} {'Vector-Only':<15} {'Hybrid':<15} {'Delta / Impact'}")
    print("-" * 80)
    for m in ["recall@1", "recall@3", "recall@5", "hit_rate@3", "mrr"]:
        vm = vec.get(m, 0.0)
        hm = hyb.get(m, 0.0)
        delta = hm - vm
        sign = "+" if delta >= 0 else ""
        print(f"  {m.upper():<25} {vm:<15.4f} {hm:<15.4f} {sign}{delta:.4f}")

    print("-" * 80)
    print(f"  {'Seed Hit Rate':<25} {'-':<15} {graph_s.get('seed_hit_rate', 0.0)*100:<14.1f}% -")
    print(f"  {'Graph/Vector Overlap':<25} {'-':<15} {graph_s.get('graph_vector_overlap_rate', 0.0)*100:<14.1f}% -")
    print(f"  {'Unique Relevant Chunks':<25} {'-':<15} {graph_s.get('total_unique_relevant_chunks_added', 0):<15} -")
    print("-" * 80)
    print(f"  {'Avg Retrieval Latency':<25} {lat.get('vector_retrieval_avg_ms', 0):<12.1f}ms {lat.get('hybrid_retrieval_avg_ms', 0):<12.1f}ms +{lat.get('retrieval_overhead_avg_ms', 0):.1f}ms overhead")
    print("=" * 80)


if __name__ == "__main__":
    main()
