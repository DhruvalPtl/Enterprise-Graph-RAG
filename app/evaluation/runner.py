"""
Benchmark Evaluation Runner for Graph RAG (Phase G5).
Executes side-by-side evaluation between Vector-Only RAG (System A) and
Hybrid Graph + Vector RAG (System B) under strictly identical conditions.
"""
import logging
import time
from typing import Any, Dict, List, Optional
import psycopg

from app.db import get_connection
from app.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from app.evaluation.metrics import (
    calculate_retrieval_metrics,
    calculate_graph_metrics,
    calculate_citation_metrics,
    calculate_generation_metrics,
)
from app.hybrid_retriever import GraphVectorHybridRetriever
from app.models import AccessContext, RAGResponse, SearchResult
from app.reranker import RerankedRetrievalPipeline
from app.rag import RAGPipeline

logger = logging.getLogger("evaluation_runner")


class EvaluationRunner:
    """
    Coordinates side-by-side evaluation of Vector-Only and Hybrid Graph+Vector RAG systems.
    Captures complete auditable traces including candidate lists, provenance,
    latency breakdowns, and deterministic performance metrics.
    """

    def __init__(
        self,
        candidate_k: int = 10,
        top_k: int = 3,
        temperature: float = 0.0,
        enable_llm: bool = True,
        vector_pipeline: Optional[RAGPipeline] = None,
        hybrid_pipeline: Optional[RAGPipeline] = None,
    ):
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.temperature = temperature
        self.enable_llm = enable_llm

        # Dependency injection of custom/mock pipelines for testing
        self.vector_pipeline = vector_pipeline
        self.hybrid_pipeline = hybrid_pipeline

    def _get_pipelines(self) -> tuple[RAGPipeline, RAGPipeline]:
        """Initializes or returns both RAG pipelines."""
        vec_pipe = self.vector_pipeline or RAGPipeline(
            enable_graph=False,
            candidate_k=self.candidate_k,
            top_k=self.top_k,
        )
        hyb_pipe = self.hybrid_pipeline or RAGPipeline(
            enable_graph=True,
            candidate_k=self.candidate_k,
            top_k=self.top_k,
        )
        return vec_pipe, hyb_pipe

    def evaluate_question(
        self,
        question: EvaluationQuestion,
        conn: Optional[psycopg.Connection] = None,
        access_context: Optional[AccessContext] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single evaluation question across both System A (Vector-only)
        and System B (Hybrid Graph + Vector), computing all metrics and recording traces.
        """
        ctx = access_context or AccessContext()
        vec_pipe, hyb_pipe = self._get_pipelines()
        gold_chunks = question.expected_source_chunks

        # ======================================================================
        # 1. SYSTEM A: VECTOR-ONLY RAG
        # ======================================================================
        t0 = time.time()
        retrieval_t0 = time.time()

        # Retrieve candidates via vector/BM25/RRF pipeline
        vec_retriever: RerankedRetrievalPipeline = vec_pipe.retrieval_pipeline
        vec_diag = vec_retriever.retrieve_with_diagnostics(
            query_text=question.question,
            candidate_k=self.candidate_k,
            top_k=self.top_k,
            access_context=ctx,
            conn=conn,
        )
        vec_retrieval_ms = int((time.time() - retrieval_t0) * 1000)

        vec_reranked: List[SearchResult] = vec_diag.get("reranked_results", [])
        vec_retrieved_chunk_ids = [r.chunk_id for r in vec_reranked]
        vec_candidate_chunk_ids = [
            (c.get("chunk_id", c.get("id")) if isinstance(c, dict) else getattr(c, "chunk_id", None))
            for c in vec_diag.get("candidates", [])
        ]

        vec_answer = ""
        vec_citations: List[Dict[str, Any]] = []
        vec_gen_ms = 0

        if self.enable_llm:
            gen_t0 = time.time()
            vec_response: RAGResponse = vec_pipe.answer_query(
                query=question.question,
                candidate_k=self.candidate_k,
                top_k=self.top_k,
                access_context=ctx,
                conn=conn,
                temperature=self.temperature,
            )
            vec_gen_ms = int((time.time() - gen_t0) * 1000)
            vec_answer = vec_response.answer
            vec_citations = [c.to_dict() if hasattr(c, "to_dict") else (c.model_dump() if hasattr(c, "model_dump") else dict(c)) for c in vec_response.citations]

        vec_total_ms = int((time.time() - t0) * 1000)

        # Vector-only metrics
        vec_ret_metrics = calculate_retrieval_metrics(vec_retrieved_chunk_ids, gold_chunks)
        vec_cite_chunk_ids = [c["chunk_id"] for c in vec_citations]
        vec_cite_metrics = calculate_citation_metrics(vec_cite_chunk_ids, gold_chunks)
        vec_gen_metrics = calculate_generation_metrics(
            generated_answer=vec_answer,
            expected_answer=question.expected_answer,
            expected_entities=question.expected_entities,
            retrieved_chunk_contents=[r.content for r in vec_reranked],
            is_no_evidence_query=question.is_no_evidence,
        )

        # ======================================================================
        # 2. SYSTEM B: HYBRID GRAPH + VECTOR RAG
        # ======================================================================
        t0_hyb = time.time()
        retrieval_hyb_t0 = time.time()

        hyb_retriever: GraphVectorHybridRetriever = hyb_pipe.retrieval_pipeline
        hyb_result = hyb_retriever.retrieve_fused(
            query=question.question,
            candidate_k=self.candidate_k,
            top_k=self.top_k,
            access_context=ctx,
            conn=conn,
        )
        hyb_retrieval_ms = int((time.time() - retrieval_hyb_t0) * 1000)

        hyb_reranked: List[SearchResult] = hyb_result.reranked_results
        hyb_retrieved_chunk_ids = [r.chunk_id for r in hyb_reranked]
        hyb_vec_candidate_ids = [c.chunk_id for c in hyb_result.vector_candidates]
        hyb_graph_candidate_ids = [c.chunk_id for c in hyb_result.graph_candidates]
        hyb_fused_candidate_ids = [c.chunk_id for c in hyb_result.fused_candidates]

        hyb_answer = ""
        hyb_citations: List[Dict[str, Any]] = []
        hyb_gen_ms = 0

        if self.enable_llm:
            gen_t0 = time.time()
            hyb_response: RAGResponse = hyb_pipe.answer_query(
                query=question.question,
                candidate_k=self.candidate_k,
                top_k=self.top_k,
                access_context=ctx,
                conn=conn,
                temperature=self.temperature,
            )
            hyb_gen_ms = int((time.time() - gen_t0) * 1000)
            hyb_answer = hyb_response.answer
            hyb_citations = [c.to_dict() if hasattr(c, "to_dict") else (c.model_dump() if hasattr(c, "model_dump") else dict(c)) for c in hyb_response.citations]

        hyb_total_ms = int((time.time() - t0_hyb) * 1000)

        # Hybrid metrics
        hyb_ret_metrics = calculate_retrieval_metrics(hyb_retrieved_chunk_ids, gold_chunks)
        hyb_cite_chunk_ids = [c["chunk_id"] for c in hyb_citations]
        hyb_cite_metrics = calculate_citation_metrics(hyb_cite_chunk_ids, gold_chunks)
        hyb_gen_metrics = calculate_generation_metrics(
            generated_answer=hyb_answer,
            expected_answer=question.expected_answer,
            expected_entities=question.expected_entities,
            retrieved_chunk_contents=[r.content for r in hyb_reranked],
            is_no_evidence_query=question.is_no_evidence,
        )

        # Graph-specific metrics
        graph_relationships = []
        matched_seeds = 0
        if hyb_result.graph_result:
            matched_seeds = len(hyb_result.graph_result.matched_entities)
            for r in hyb_result.graph_result.relationships:
                graph_relationships.append({
                    "source": r.metadata.get("source_name", ""),
                    "type": r.relationship_type,
                    "target": r.metadata.get("target_name", ""),
                    "chunk_id": r.chunk_id,
                })

        graph_metrics = calculate_graph_metrics(
            vector_candidate_chunk_ids=hyb_vec_candidate_ids,
            graph_candidate_chunk_ids=hyb_graph_candidate_ids,
            expected_chunk_ids=gold_chunks,
            matched_seed_count=matched_seeds,
            retrieved_relationships=graph_relationships,
            expected_relationships=question.expected_relationships,
        )

        # ======================================================================
        # 3. COMPARISON & ERROR ANALYSIS
        # ======================================================================
        vec_r3 = vec_ret_metrics["recall@3"]
        hyb_r3 = hyb_ret_metrics["recall@3"]
        unique_rel_count = graph_metrics["unique_relevant_graph_chunks_count"]

        if hyb_r3 > vec_r3 or unique_rel_count > 0:
            verdict = "hybrid_won"
            analysis = (
                f"Graph retrieval added {unique_rel_count} unique relevant evidence chunks "
                f"(Recall@3: {hyb_r3:.2f} vs {vec_r3:.2f})."
            )
        elif vec_r3 > hyb_r3:
            verdict = "vector_won"
            analysis = (
                f"Vector-only achieved higher top-K recall ({vec_r3:.2f} vs {hyb_r3:.2f}); "
                f"graph candidates were either down-ranked or less relevant."
            )
        else:
            verdict = "tie"
            if len(gold_chunks) == 0:
                analysis = "Negative control: both systems correctly handled absence of evidence."
            elif graph_metrics["graph_overlap_count"] == len(hyb_graph_candidate_ids):
                analysis = "Tie: Graph retrieved evidence that completely overlapped with vector candidates."
            else:
                analysis = f"Tie: Both systems achieved identical Recall@3 ({vec_r3:.2f})."

        return {
            "question_id": question.id,
            "question": question.question,
            "category": question.category,
            "expected_source_chunks": gold_chunks,
            "verdict": verdict,
            "analysis": analysis,
            "vector_only": {
                "candidate_chunk_ids": vec_candidate_chunk_ids,
                "retrieved_chunk_ids": vec_retrieved_chunk_ids,
                "retrieval_metrics": vec_ret_metrics,
                "citation_metrics": vec_cite_metrics,
                "generation_metrics": vec_gen_metrics,
                "timing": {
                    "retrieval_ms": vec_retrieval_ms,
                    "generation_ms": vec_gen_ms,
                    "total_ms": vec_total_ms,
                },
                "answer": vec_answer,
                "citations": vec_citations,
            },
            "hybrid": {
                "vector_candidate_chunk_ids": hyb_vec_candidate_ids,
                "graph_candidate_chunk_ids": hyb_graph_candidate_ids,
                "fused_candidate_chunk_ids": hyb_fused_candidate_ids,
                "retrieved_chunk_ids": hyb_retrieved_chunk_ids,
                "graph_relationships": graph_relationships,
                "retrieval_metrics": hyb_ret_metrics,
                "graph_metrics": graph_metrics,
                "citation_metrics": hyb_cite_metrics,
                "generation_metrics": hyb_gen_metrics,
                "timing": {
                    "vector_retrieval_ms": hyb_result.execution_timing_ms.get("vector_retrieval_ms", 0),
                    "graph_retrieval_ms": hyb_result.execution_timing_ms.get("graph_retrieval_ms", 0),
                    "fusion_ms": hyb_result.execution_timing_ms.get("fusion_ms", 0),
                    "rerank_ms": hyb_result.execution_timing_ms.get("rerank_ms", 0),
                    "retrieval_total_ms": hyb_retrieval_ms,
                    "generation_ms": hyb_gen_ms,
                    "total_ms": hyb_total_ms,
                },
                "answer": hyb_answer,
                "citations": hyb_citations,
            },
            "retrieval_overhead_ms": max(0, hyb_retrieval_ms - vec_retrieval_ms),
        }

    def run_benchmark(
        self,
        dataset: EvaluationDataset,
        limit: Optional[int] = None,
        category: Optional[str] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full evaluation benchmark across all questions in the dataset.
        Aggregates overall and category-level metrics and produces auditable trace payloads.
        """
        questions = dataset.questions
        if category:
            questions = dataset.filter_by_category(category)
        if limit:
            questions = questions[:limit]

        results: List[Dict[str, Any]] = []
        logger.info(f"Starting evaluation benchmark for {len(questions)} questions...")

        db_conn = conn or get_connection()
        should_close_conn = conn is None

        try:
            for idx, q in enumerate(questions, start=1):
                logger.info(f"Evaluating [{idx}/{len(questions)}] {q.id} ({q.category}): {q.question[:60]}...")
                res = self.evaluate_question(q, conn=db_conn)
                results.append(res)
        finally:
            if should_close_conn:
                db_conn.close()

        summary = self._aggregate_summary(results)
        return {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_questions": len(results),
                "candidate_k": self.candidate_k,
                "top_k": self.top_k,
                "enable_llm": self.enable_llm,
                "temperature": self.temperature,
            },
            "summary": summary,
            "traces": results,
        }

    def _aggregate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates individual question traces into overall and category metrics."""
        if not results:
            return {}

        total = len(results)
        categories = sorted(list(set(r["category"] for r in results)))

        def mean(values: List[float]) -> float:
            return round(sum(values) / max(1, len(values)), 4)

        # Overall retrieval
        vec_r1 = mean([r["vector_only"]["retrieval_metrics"]["recall@1"] for r in results])
        vec_r3 = mean([r["vector_only"]["retrieval_metrics"]["recall@3"] for r in results])
        vec_r5 = mean([r["vector_only"]["retrieval_metrics"]["recall@5"] for r in results])
        vec_hit3 = mean([r["vector_only"]["retrieval_metrics"]["hit_rate@3"] for r in results])
        vec_mrr = mean([r["vector_only"]["retrieval_metrics"]["mrr"] for r in results])

        hyb_r1 = mean([r["hybrid"]["retrieval_metrics"]["recall@1"] for r in results])
        hyb_r3 = mean([r["hybrid"]["retrieval_metrics"]["recall@3"] for r in results])
        hyb_r5 = mean([r["hybrid"]["retrieval_metrics"]["recall@5"] for r in results])
        hyb_hit3 = mean([r["hybrid"]["retrieval_metrics"]["hit_rate@3"] for r in results])
        hyb_mrr = mean([r["hybrid"]["retrieval_metrics"]["mrr"] for r in results])

        # Graph-specific
        seed_hit = mean([r["hybrid"]["graph_metrics"]["graph_seed_hit"] for r in results])
        rel_rate = mean([r["hybrid"]["graph_metrics"]["relationship_retrieval_rate"] for r in results])
        prov_cov = mean([r["hybrid"]["graph_metrics"]["graph_provenance_coverage"] for r in results])
        overlap = mean([r["hybrid"]["graph_metrics"]["graph_vector_overlap_rate"] for r in results])
        total_unique_rel_chunks = sum(r["hybrid"]["graph_metrics"]["unique_relevant_graph_chunks_count"] for r in results)

        # Generation & Citation
        vec_cite_corr = mean([r["vector_only"]["citation_metrics"]["citation_correctness"] for r in results])
        vec_cite_comp = mean([r["vector_only"]["citation_metrics"]["citation_completeness"] for r in results])
        vec_ans_corr = mean([r["vector_only"]["generation_metrics"]["answer_correctness"] for r in results])
        vec_ground = mean([r["vector_only"]["generation_metrics"]["groundedness_score"] for r in results])

        hyb_cite_corr = mean([r["hybrid"]["citation_metrics"]["citation_correctness"] for r in results])
        hyb_cite_comp = mean([r["hybrid"]["citation_metrics"]["citation_completeness"] for r in results])
        hyb_ans_corr = mean([r["hybrid"]["generation_metrics"]["answer_correctness"] for r in results])
        hyb_ground = mean([r["hybrid"]["generation_metrics"]["groundedness_score"] for r in results])

        # Latencies
        vec_ret_lat = mean([r["vector_only"]["timing"]["retrieval_ms"] for r in results])
        vec_tot_lat = mean([r["vector_only"]["timing"]["total_ms"] for r in results])
        hyb_ret_lat = mean([r["hybrid"]["timing"]["retrieval_total_ms"] for r in results])
        hyb_tot_lat = mean([r["hybrid"]["timing"]["total_ms"] for r in results])
        avg_overhead = mean([r["retrieval_overhead_ms"] for r in results])

        # Category breakdown
        category_breakdown = {}
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            category_breakdown[cat] = {
                "count": len(cat_results),
                "vector_recall@3": mean([r["vector_only"]["retrieval_metrics"]["recall@3"] for r in cat_results]),
                "hybrid_recall@3": mean([r["hybrid"]["retrieval_metrics"]["recall@3"] for r in cat_results]),
                "vector_hit@3": mean([r["vector_only"]["retrieval_metrics"]["hit_rate@3"] for r in cat_results]),
                "hybrid_hit@3": mean([r["hybrid"]["retrieval_metrics"]["hit_rate@3"] for r in cat_results]),
                "vector_mrr": mean([r["vector_only"]["retrieval_metrics"]["mrr"] for r in cat_results]),
                "hybrid_mrr": mean([r["hybrid"]["retrieval_metrics"]["mrr"] for r in cat_results]),
                "unique_relevant_graph_chunks": sum(r["hybrid"]["graph_metrics"]["unique_relevant_graph_chunks_count"] for r in cat_results),
                "verdicts": {
                    "hybrid_won": sum(1 for r in cat_results if r["verdict"] == "hybrid_won"),
                    "vector_won": sum(1 for r in cat_results if r["verdict"] == "vector_won"),
                    "tie": sum(1 for r in cat_results if r["verdict"] == "tie"),
                }
            }

        verdict_counts = {
            "hybrid_won": sum(1 for r in results if r["verdict"] == "hybrid_won"),
            "vector_won": sum(1 for r in results if r["verdict"] == "vector_won"),
            "tie": sum(1 for r in results if r["verdict"] == "tie"),
        }

        return {
            "overall_verdicts": verdict_counts,
            "retrieval": {
                "vector_only": {
                    "recall@1": vec_r1,
                    "recall@3": vec_r3,
                    "recall@5": vec_r5,
                    "hit_rate@3": vec_hit3,
                    "mrr": vec_mrr,
                },
                "hybrid": {
                    "recall@1": hyb_r1,
                    "recall@3": hyb_r3,
                    "recall@5": hyb_r5,
                    "hit_rate@3": hyb_hit3,
                    "mrr": hyb_mrr,
                },
            },
            "graph_specific": {
                "seed_hit_rate": seed_hit,
                "relationship_retrieval_rate": rel_rate,
                "graph_provenance_coverage": prov_cov,
                "graph_vector_overlap_rate": overlap,
                "total_unique_relevant_chunks_added": total_unique_rel_chunks,
            },
            "generation": {
                "vector_only": {
                    "citation_correctness": vec_cite_corr,
                    "citation_completeness": vec_cite_comp,
                    "answer_correctness": vec_ans_corr,
                    "groundedness": vec_ground,
                },
                "hybrid": {
                    "citation_correctness": hyb_cite_corr,
                    "citation_completeness": hyb_cite_comp,
                    "answer_correctness": hyb_ans_corr,
                    "groundedness": hyb_ground,
                },
            },
            "latency": {
                "vector_retrieval_avg_ms": vec_ret_lat,
                "vector_total_avg_ms": vec_tot_lat,
                "hybrid_retrieval_avg_ms": hyb_ret_lat,
                "hybrid_total_avg_ms": hyb_tot_lat,
                "retrieval_overhead_avg_ms": avg_overhead,
            },
            "category_breakdown": category_breakdown,
        }
