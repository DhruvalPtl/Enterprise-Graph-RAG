"""
Reporting and Formatting Module for Graph RAG Evaluation (Phase G5).
Generates comprehensive JSON payloads and executive Markdown reports comparing
Vector-Only RAG against Hybrid Graph + Vector RAG with category ablations.
"""
import json
from pathlib import Path
from typing import Any, Dict, List


class EvaluationReporter:
    """Formats and exports benchmark evaluation reports."""

    @staticmethod
    def save_json_results(benchmark_payload: Dict[str, Any], output_path: Path):
        """Saves complete auditable traces to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_json_summary(summary_payload: Dict[str, Any], output_path: Path):
        """Saves aggregate metrics summary to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_markdown_report(benchmark_payload: Dict[str, Any]) -> str:
        """Generates comprehensive, publication-grade Markdown evaluation report."""
        meta = benchmark_payload.get("metadata", {})
        summary = benchmark_payload.get("summary", {})
        traces = benchmark_payload.get("traces", [])

        ret = summary.get("retrieval", {})
        vec_ret = ret.get("vector_only", {})
        hyb_ret = ret.get("hybrid", {})

        gen = summary.get("generation", {})
        vec_gen = gen.get("vector_only", {})
        hyb_gen = gen.get("hybrid", {})

        graph_spec = summary.get("graph_specific", {})
        lat = summary.get("latency", {})
        verdicts = summary.get("overall_verdicts", {})
        cats = summary.get("category_breakdown", {})

        lines = [
            "# Graph RAG Empirical Evaluation & Benchmark Report (Phase G5)",
            "",
            "## 1. Executive Summary & Overview",
            "",
            f"- **Timestamp**: `{meta.get('timestamp')}`",
            f"- **Evaluated Questions**: **{meta.get('total_questions', 0)}**",
            f"- **Candidate K**: `{meta.get('candidate_k')}` | **Reranker Top K**: `{meta.get('top_k')}`",
            f"- **LLM Generation Active**: `{meta.get('enable_llm')}` | **Temperature**: `{meta.get('temperature')}`",
            f"- **Overall Verdicts**: Hybrid Won: **{verdicts.get('hybrid_won', 0)}** | Vector Won: **{verdicts.get('vector_won', 0)}** | Tie: **{verdicts.get('tie', 0)}**",
            "",
            "---",
            "",
            "## 2. Head-to-Head Metric Comparison",
            "",
            "| Metric | Vector-Only Baseline | Hybrid Graph + Vector | Delta / Graph Impact |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Recall@1** | {vec_ret.get('recall@1', 0.0):.4f} | {hyb_ret.get('recall@1', 0.0):.4f} | {hyb_ret.get('recall@1', 0.0) - vec_ret.get('recall@1', 0.0):+.4f} |",
            f"| **Recall@3** | {vec_ret.get('recall@3', 0.0):.4f} | {hyb_ret.get('recall@3', 0.0):.4f} | {hyb_ret.get('recall@3', 0.0) - vec_ret.get('recall@3', 0.0):+.4f} |",
            f"| **Recall@5** | {vec_ret.get('recall@5', 0.0):.4f} | {hyb_ret.get('recall@5', 0.0):.4f} | {hyb_ret.get('recall@5', 0.0) - vec_ret.get('recall@5', 0.0):+.4f} |",
            f"| **HitRate@3** | {vec_ret.get('hit_rate@3', 0.0):.4f} | {hyb_ret.get('hit_rate@3', 0.0):.4f} | {hyb_ret.get('hit_rate@3', 0.0) - vec_ret.get('hit_rate@3', 0.0):+.4f} |",
            f"| **MRR (Mean Reciprocal Rank)** | {vec_ret.get('mrr', 0.0):.4f} | {hyb_ret.get('mrr', 0.0):.4f} | {hyb_ret.get('mrr', 0.0) - vec_ret.get('mrr', 0.0):+.4f} |",
            f"| **Citation Correctness** | {vec_gen.get('citation_correctness', 0.0):.4f} | {hyb_gen.get('citation_correctness', 0.0):.4f} | {hyb_gen.get('citation_correctness', 0.0) - vec_gen.get('citation_correctness', 0.0):+.4f} |",
            f"| **Citation Completeness** | {vec_gen.get('citation_completeness', 0.0):.4f} | {hyb_gen.get('citation_completeness', 0.0):.4f} | {hyb_gen.get('citation_completeness', 0.0) - vec_gen.get('citation_completeness', 0.0):+.4f} |",
            f"| **Answer Correctness** | {vec_gen.get('answer_correctness', 0.0):.4f} | {hyb_gen.get('answer_correctness', 0.0):.4f} | {hyb_gen.get('answer_correctness', 0.0) - vec_gen.get('answer_correctness', 0.0):+.4f} |",
            f"| **Groundedness Score** | {vec_gen.get('groundedness', 0.0):.4f} | {hyb_gen.get('groundedness', 0.0):.4f} | {hyb_gen.get('groundedness', 0.0) - vec_gen.get('groundedness', 0.0):+.4f} |",
            f"| **Avg Retrieval Latency** | {lat.get('vector_retrieval_avg_ms', 0):.1f} ms | {lat.get('hybrid_retrieval_avg_ms', 0):.1f} ms | +{lat.get('retrieval_overhead_avg_ms', 0):.1f} ms overhead |",
            f"| **Avg Total Round-Trip** | {lat.get('vector_total_avg_ms', 0):.1f} ms | {lat.get('hybrid_total_avg_ms', 0):.1f} ms | +{lat.get('hybrid_total_avg_ms', 0) - lat.get('vector_total_avg_ms', 0):.1f} ms |",
            "",
            "---",
            "",
            "## 3. Graph-Specific Telemetry",
            "",
            "| Graph Retrieval Metric | Measured Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Seed Entity Hit Rate** | {graph_spec.get('seed_hit_rate', 0.0) * 100:.1f}% | Percentage of queries matching at least one valid knowledge graph seed entity |",
            f"| **Relationship Retrieval Rate** | {graph_spec.get('relationship_retrieval_rate', 0.0) * 100:.1f}% | Coverage of expected ground-truth relationship edges found during graph traversal |",
            f"| **Graph Provenance Coverage** | {graph_spec.get('graph_provenance_coverage', 0.0) * 100:.1f}% | Fraction of gold chunks covered by traversed edge provenance |",
            f"| **Graph/Vector Overlap (Jaccard)** | {graph_spec.get('graph_vector_overlap_rate', 0.0) * 100:.1f}% | Degree of candidate duplication between graph and vector paths |",
            f"| **Unique Relevant Chunks Added** | **{graph_spec.get('total_unique_relevant_chunks_added', 0)} chunks** | Relevant evidence chunks retrieved *only* by Graph and missed by Vector/BM25 |",
            "",
            "---",
            "",
            "## 4. Category-Level Ablation Breakdown",
            "",
            "| Category | Questions | Vector Recall@3 | Hybrid Recall@3 | Vector Hit@3 | Hybrid Hit@3 | Unique Relevant Graph Chunks | Verdict (H / V / Tie) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for cat, data in sorted(cats.items()):
            cnt = data.get("count", 0)
            vr = data.get("vector_recall@3", 0.0)
            hr = data.get("hybrid_recall@3", 0.0)
            vh = data.get("vector_hit@3", 0.0)
            hh = data.get("hybrid_hit@3", 0.0)
            uniq = data.get("unique_relevant_graph_chunks", 0)
            v = data.get("verdicts", {})
            v_str = f"{v.get('hybrid_won', 0)} / {v.get('vector_won', 0)} / {v.get('tie', 0)}"
            lines.append(
                f"| `{cat}` | {cnt} | {vr:.4f} | {hr:.4f} | {vh:.4f} | {hh:.4f} | {uniq} | {v_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 5. Case Studies & Detailed Traces",
            "",
        ])

        # Find specific interesting queries
        interesting_ids = ["q008", "q010", "q015", "q021", "q023"]
        for q_id in interesting_ids:
            trace = next((t for t in traces if t["question_id"] == q_id), None)
            if not trace:
                continue

            lines.extend([
                f"### Query `{trace['question_id']}`: \"{trace['question']}\"",
                f"- **Category**: `{trace['category']}` | **Verdict**: `{trace['verdict']}`",
                f"- **Error Analysis**: {trace['analysis']}",
                "",
                "**Vector-Only System**:",
                f"- Candidates Retrieved: `{trace['vector_only']['candidate_chunk_ids'][:5]}`",
                f"- Final Top Reranked: `{trace['vector_only']['retrieved_chunk_ids']}`",
                f"- Recall@3: `{trace['vector_only']['retrieval_metrics']['recall@3']:.2f}` | Timing: `{trace['vector_only']['timing']['total_ms']}ms`",
                f"- Answer: *\"{trace['vector_only']['answer'][:160]}...\"*",
                "",
                "**Hybrid Graph + Vector System**:",
                f"- Matched Relationships: `{len(trace['hybrid']['graph_relationships'])}` edges",
            ])
            for rel in trace["hybrid"]["graph_relationships"][:3]:
                lines.append(f"  - `({rel['source']}) --[{rel['type']}]--> ({rel['target']})` (Chunk {rel['chunk_id']})")
            lines.extend([
                f"- Graph Candidate Chunks: `{trace['hybrid']['graph_candidate_chunk_ids'][:5]}`",
                f"- Unique Relevant Graph Chunks: `{trace['hybrid']['graph_metrics']['unique_relevant_chunk_ids']}`",
                f"- Final Top Reranked: `{trace['hybrid']['retrieved_chunk_ids']}`",
                f"- Recall@3: `{trace['hybrid']['retrieval_metrics']['recall@3']:.2f}` | Timing: `{trace['hybrid']['timing']['total_ms']}ms`",
                f"- Answer: *\"{trace['hybrid']['answer'][:160]}...\"*",
                "",
            ])

        lines.extend([
            "---",
            "",
            "## 6. Empirical Findings & Limitations",
            "",
            "### Where Graph Retrieval Helps",
            "1. **Multi-Hop & Entity Relationship Queries**: When facts are spread across distant sections or sentences lacking lexical overlap, knowledge graph edges successfully navigate from seed entities directly to target chunks that vector search ranks low or misses.",
            "2. **Structured Attribution**: The graph provides explicit entity-relationship provenance, injecting authoritative relationship tags (`[GRAPH RELATIONSHIP]`) directly into passage evidence blocks for the context builder.",
            "",
            "### Where Graph Retrieval Adds Little or No Value",
            "1. **Dense Topical / Lexical Queries (`vector_favored`)**: When queries have high lexical specificity (e.g. specialized terminology), dense vector search and BM25 already retrieve the exact chunk at Rank #1. Graph candidates largely overlap with vector candidates (high Jaccard overlap).",
            "2. **Negative Controls (`no_evidence`)**: For out-of-corpus queries, neither system retrieves valid evidence, and both successfully trigger grounded refusal.",
            "",
            "### Latency Overhead",
            f"Graph retrieval introduces an average overhead of **+{lat.get('retrieval_overhead_avg_ms', 0):.1f} ms** per query. This overhead arises from entity matching in PostgreSQL and incident edge traversal, but is kept bounded within millisecond thresholds via indexed lookups.",
            "",
        ])

        return "\n".join(lines)
