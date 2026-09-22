"""
Unit tests for Graph RAG Phase G5 Evaluation and Benchmarking framework.
Tests dataset loading, deterministic metrics calculation, auditable runner execution,
and report generation with zero live API or network dependencies.
"""
import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from app.evaluation.dataset import (
    EvaluationDataset,
    EvaluationQuestion,
    ExpectedRelationship,
    load_evaluation_dataset,
    DEFAULT_EVALUATION_DATASET_PATH,
)
from app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_retrieval_metrics,
    calculate_graph_metrics,
    calculate_citation_metrics,
    calculate_generation_metrics,
)
from app.evaluation.runner import EvaluationRunner
from app.evaluation.reporting import EvaluationReporter
from app.models import RAGResponse, SearchResult, Citation, AccessContext


# ==============================================================================
# 1. DATASET LOADING & SCHEMA VALIDATION TESTS
# ==============================================================================

def test_evaluation_dataset_loading_and_validation():
    """Verify that the gold dataset loads correctly and validates against Pydantic schema."""
    assert DEFAULT_EVALUATION_DATASET_PATH.exists(), "Gold evaluation dataset file must exist."
    dataset = load_evaluation_dataset(DEFAULT_EVALUATION_DATASET_PATH)

    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset) >= 20, f"Expected at least 20 evaluation questions, got {len(dataset)}"
    assert len(dataset.corpus_documents) >= 2

    # Check categories coverage
    categories = dataset.categories
    expected_cats = {
        "direct_factual",
        "entity_relationship",
        "multi_hop_relationship",
        "comparison",
        "multiple_source_chunks",
        "vector_favored",
        "graph_favored",
        "no_evidence",
    }
    assert expected_cats.issubset(categories), f"Missing categories: {expected_cats - categories}"


def test_evaluation_dataset_methods():
    """Test helper methods on EvaluationDataset."""
    dataset = load_evaluation_dataset(DEFAULT_EVALUATION_DATASET_PATH)

    q = dataset.get_by_id("q001")
    assert q is not None
    assert q.id == "q001"
    assert q.category == "direct_factual"

    no_ev = dataset.filter_by_category("no_evidence")
    assert len(no_ev) >= 2
    for item in no_ev:
        assert item.is_no_evidence


# ==============================================================================
# 2. DETERMINISTIC RETRIEVAL METRICS TESTS
# ==============================================================================

def test_recall_at_k_calculation():
    """Verify Recall@K calculation under perfect, partial, and zero recall."""
    gold = [101, 102, 103, 104]

    # Perfect recall in top 4
    retrieved_perfect = [101, 102, 103, 104, 999]
    assert calculate_recall_at_k(retrieved_perfect, gold, k=4) == 1.0

    # Partial recall (2 out of 4)
    retrieved_partial = [101, 999, 102, 888]
    assert calculate_recall_at_k(retrieved_partial, gold, k=4) == 0.5

    # Zero recall
    retrieved_zero = [999, 888, 777]
    assert calculate_recall_at_k(retrieved_zero, gold, k=3) == 0.0

    # K cutoff boundary
    retrieved_boundary = [999, 999, 101, 102]
    assert calculate_recall_at_k(retrieved_boundary, gold, k=2) == 0.0
    assert calculate_recall_at_k(retrieved_boundary, gold, k=3) == 0.25


def test_hit_rate_at_k_calculation():
    """Verify HitRate@K binary indicator."""
    gold = [10, 20]
    assert calculate_hit_rate_at_k([10, 30, 40], gold, k=1) == 1.0
    assert calculate_hit_rate_at_k([99, 10, 40], gold, k=1) == 0.0
    assert calculate_hit_rate_at_k([99, 10, 40], gold, k=2) == 1.0


def test_mrr_calculation():
    """Verify Mean Reciprocal Rank (1/rank) for first hit."""
    gold = [42]
    assert calculate_mrr([42, 1, 2], gold) == 1.0
    assert calculate_mrr([1, 42, 2], gold) == 0.5
    assert calculate_mrr([1, 2, 42], gold) == 1.0 / 3.0
    assert calculate_mrr([1, 2, 3], gold) == 0.0


def test_empty_no_evidence_retrieval_metrics():
    """Verify negative control behavior when expected gold chunks are empty."""
    assert calculate_recall_at_k([], [], k=5) == 1.0
    assert calculate_recall_at_k([10, 20], [], k=5) == 0.0
    assert calculate_hit_rate_at_k([], [], k=5) == 1.0
    assert calculate_hit_rate_at_k([10], [], k=5) == 0.0
    assert calculate_mrr([], []) == 1.0


# ==============================================================================
# 3. GRAPH-SPECIFIC METRICS TESTS
# ==============================================================================

def test_graph_specific_metrics():
    """Verify graph seed hit rate, relationship retrieval rate, overlap, and unique evidence."""
    vec_chunks = [1, 2, 3, 4]
    graph_chunks = [3, 4, 5, 6]
    expected_gold = [5, 6, 7]

    expected_rels = [
        ExpectedRelationship(source="Gemini Ultra", type="DEVELOPED_BY", target="Google"),
        ExpectedRelationship(source="Gemini Ultra", type="EVALUATED_ON", target="MMLU"),
    ]
    retrieved_rels = [
        {"source_name": "Gemini Ultra", "relationship_type": "DEVELOPED_BY", "target_name": "Google", "chunk_id": 5}
    ]

    metrics = calculate_graph_metrics(
        vector_candidate_chunk_ids=vec_chunks,
        graph_candidate_chunk_ids=graph_chunks,
        expected_chunk_ids=expected_gold,
        matched_seed_count=2,
        retrieved_relationships=retrieved_rels,
        expected_relationships=expected_rels,
    )

    assert metrics["graph_seed_hit"] == 1.0
    assert metrics["relationship_retrieval_rate"] == 0.5  # 1 out of 2 matched
    assert metrics["unique_graph_chunks_count"] == 2      # chunks 5 and 6
    assert metrics["unique_relevant_graph_chunks_count"] == 2  # chunks 5 and 6 are in gold
    assert metrics["unique_relevant_chunk_ids"] == [5, 6]
    assert metrics["graph_overlap_count"] == 2            # chunks 3 and 4 overlap


# ==============================================================================
# 4. CITATION & GENERATION METRICS TESTS
# ==============================================================================

def test_citation_metrics():
    """Verify citation correctness and completeness."""
    gold = [10, 20, 30]
    cited = [10, 20, 99]  # 2 correct out of 3 cited, 2 out of 3 gold covered

    metrics = calculate_citation_metrics(cited, gold)
    assert round(metrics["citation_correctness"], 4) == round(2.0 / 3.0, 4)
    assert round(metrics["citation_completeness"], 4) == round(2.0 / 3.0, 4)


def test_generation_metrics():
    """Verify answer correctness and groundedness calculation."""
    expected_answer = "Gemini Ultra was developed by Google and evaluated on MMLU."
    expected_entities = ["Google", "Gemini Ultra", "MMLU"]
    retrieved_contents = [
        "Gemini Ultra is a large multimodal model developed by Google.",
        "It achieved state of the art results evaluated on MMLU benchmark.",
    ]

    # Full entity coverage and grounded
    gen_ans = "Google developed Gemini Ultra, which was benchmarked and evaluated on MMLU."
    metrics = calculate_generation_metrics(
        generated_answer=gen_ans,
        expected_answer=expected_answer,
        expected_entities=expected_entities,
        retrieved_chunk_contents=retrieved_contents,
        is_no_evidence_query=False,
    )
    assert metrics["entity_coverage"] == 1.0
    assert metrics["groundedness_score"] > 0.8
    assert metrics["answer_correctness"] > 0.8


def test_generation_metrics_refusal_negative_control():
    """Verify negative control detection for no_evidence queries."""
    refusal_answer = "The available documents do not provide sufficient information to answer this question."
    metrics = calculate_generation_metrics(
        generated_answer=refusal_answer,
        expected_answer="Refusal",
        expected_entities=[],
        retrieved_chunk_contents=[],
        is_no_evidence_query=True,
    )
    assert metrics["refusal_correctness"] == 1.0
    assert metrics["answer_correctness"] == 1.0


# ==============================================================================
# 5. RUNNER & PIPELINE EVALUATION INTEGRATION TESTS (MOCKED)
# ==============================================================================

def test_evaluation_runner_with_mocked_pipelines():
    """Test EvaluationRunner executing a question with mock Vector and Hybrid pipelines."""
    # 1. Mock SearchResults
    sr1 = SearchResult(chunk_id=41107, document_id="91", content="Gemini Ultra developed by Google", score=0.9, rank=1)
    sr2 = SearchResult(chunk_id=41101, document_id="91", content="Evaluated on MMLU benchmark", score=0.85, rank=2)

    # 2. Mock Vector Pipeline
    mock_vec_pipe = MagicMock()
    mock_vec_retriever = MagicMock()
    mock_vec_retriever.retrieve_with_diagnostics.return_value = {
        "candidates": [sr1],
        "reranked_results": [sr1],
    }
    mock_vec_pipe.retrieval_pipeline = mock_vec_retriever
    mock_vec_pipe.answer_query.return_value = RAGResponse(
        query="test",
        answer="Google developed Gemini.",
        citations=[Citation(source_id=1, document_id="91", chunk_id=41107, filename="doc.pdf", page_number=5)],
    )

    # 3. Mock Hybrid Pipeline
    mock_hyb_pipe = MagicMock()
    mock_hyb_retriever = MagicMock()
    mock_hybrid_res = MagicMock()
    mock_hybrid_res.vector_candidates = [sr1]
    mock_hybrid_res.graph_candidates = [sr2]
    mock_hybrid_res.fused_candidates = [sr1, sr2]
    mock_hybrid_res.reranked_results = [sr1, sr2]
    mock_hybrid_res.execution_timing_ms = {"vector_retrieval_ms": 10, "graph_retrieval_ms": 15, "fusion_ms": 2, "rerank_ms": 5}
    mock_graph_res = MagicMock()
    mock_graph_res.matched_entities = [MagicMock()]
    mock_rel = MagicMock()
    mock_rel.metadata = {"source_name": "Gemini Ultra", "target_name": "MMLU"}
    mock_rel.relationship_type = "EVALUATED_ON"
    mock_rel.chunk_id = 41101
    mock_graph_res.relationships = [mock_rel]
    mock_hybrid_res.graph_result = mock_graph_res
    mock_hyb_retriever.retrieve_fused.return_value = mock_hybrid_res
    mock_hyb_pipe.retrieval_pipeline = mock_hyb_retriever
    mock_hyb_pipe.answer_query.return_value = RAGResponse(
        query="test",
        answer="Gemini Ultra was developed by Google and evaluated on MMLU.",
        citations=[
            Citation(source_id=1, document_id="91", chunk_id=41107, filename="doc.pdf", page_number=5),
            Citation(source_id=2, document_id="91", chunk_id=41101, filename="doc.pdf", page_number=3),
        ],
    )

    runner = EvaluationRunner(
        candidate_k=5,
        top_k=2,
        vector_pipeline=mock_vec_pipe,
        hybrid_pipeline=mock_hyb_pipe,
        enable_llm=True,
    )

    question = EvaluationQuestion(
        id="test_q",
        question="What organization developed Gemini Ultra and what benchmark was it evaluated on?",
        category="multi_hop_relationship",
        expected_answer="Gemini Ultra was developed by Google and evaluated on MMLU.",
        expected_entities=["Google", "Gemini Ultra", "MMLU"],
        expected_relationships=[ExpectedRelationship(source="Gemini Ultra", type="EVALUATED_ON", target="MMLU")],
        expected_source_chunks=[41107, 41101],
    )

    trace = runner.evaluate_question(question)

    assert trace["question_id"] == "test_q"
    assert trace["verdict"] == "hybrid_won"  # Hybrid got both chunks (recall=1.0), vector only 1 (recall=0.5)
    assert trace["vector_only"]["retrieval_metrics"]["recall@3"] == 0.5
    assert trace["hybrid"]["retrieval_metrics"]["recall@3"] == 1.0
    assert trace["hybrid"]["graph_metrics"]["unique_relevant_graph_chunks_count"] == 1


# ==============================================================================
# 6. REPORT GENERATOR TESTS
# ==============================================================================

def test_evaluation_reporter_markdown_rendering():
    """Verify that EvaluationReporter renders valid Markdown tables."""
    payload = {
        "metadata": {"timestamp": "2026-09-22T00:00:00Z", "total_questions": 1, "candidate_k": 10, "top_k": 3, "enable_llm": True, "temperature": 0.0},
        "summary": {
            "overall_verdicts": {"hybrid_won": 1, "vector_won": 0, "tie": 0},
            "retrieval": {
                "vector_only": {"recall@1": 0.5, "recall@3": 0.5, "recall@5": 0.5, "hit_rate@3": 1.0, "mrr": 1.0},
                "hybrid": {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "hit_rate@3": 1.0, "mrr": 1.0},
            },
            "generation": {
                "vector_only": {"citation_correctness": 1.0, "citation_completeness": 0.5, "answer_correctness": 0.7, "groundedness": 0.8},
                "hybrid": {"citation_correctness": 1.0, "citation_completeness": 1.0, "answer_correctness": 0.95, "groundedness": 0.9},
            },
            "graph_specific": {
                "seed_hit_rate": 1.0,
                "relationship_retrieval_rate": 1.0,
                "graph_provenance_coverage": 1.0,
                "graph_vector_overlap_rate": 0.5,
                "total_unique_relevant_chunks_added": 1,
            },
            "latency": {
                "vector_retrieval_avg_ms": 25.0,
                "vector_total_avg_ms": 300.0,
                "hybrid_retrieval_avg_ms": 45.0,
                "hybrid_total_avg_ms": 320.0,
                "retrieval_overhead_avg_ms": 20.0,
            },
            "category_breakdown": {
                "multi_hop_relationship": {
                    "count": 1,
                    "vector_recall@3": 0.5,
                    "hybrid_recall@3": 1.0,
                    "vector_hit@3": 1.0,
                    "hybrid_hit@3": 1.0,
                    "vector_mrr": 1.0,
                    "hybrid_mrr": 1.0,
                    "unique_relevant_graph_chunks": 1,
                    "verdicts": {"hybrid_won": 1, "vector_won": 0, "tie": 0},
                }
            },
        },
        "traces": [
            {
                "question_id": "q008",
                "question": "What organization developed Gemini Ultra?",
                "category": "multi_hop_relationship",
                "verdict": "hybrid_won",
                "analysis": "Graph added unique evidence.",
                "vector_only": {
                    "candidate_chunk_ids": [41107],
                    "retrieved_chunk_ids": [41107],
                    "retrieval_metrics": {"recall@3": 0.5},
                    "timing": {"total_ms": 300},
                    "answer": "Google developed Gemini.",
                },
                "hybrid": {
                    "graph_relationships": [{"source": "Gemini Ultra", "type": "DEVELOPED_BY", "target": "Google", "chunk_id": 41107}],
                    "graph_candidate_chunk_ids": [41101],
                    "graph_metrics": {"unique_relevant_chunk_ids": [41101]},
                    "retrieved_chunk_ids": [41107, 41101],
                    "retrieval_metrics": {"recall@3": 1.0},
                    "timing": {"total_ms": 320},
                    "answer": "Gemini Ultra was developed by Google and evaluated on MMLU.",
                },
            }
        ],
    }

    report = EvaluationReporter.generate_markdown_report(payload)
    assert "# Graph RAG Empirical Evaluation & Benchmark Report" in report
    assert "Head-to-Head Metric Comparison" in report
    assert "Category-Level Ablation Breakdown" in report
    assert "Recall@3" in report
