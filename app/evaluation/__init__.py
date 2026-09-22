"""
Evaluation and Benchmarking Module for Graph RAG (Phase G5).
Provides dataset loading, deterministic retrieval/generation metrics,
auditable evaluation runners, and comparative reporting.
"""

from app.evaluation.dataset import (
    EvaluationDataset,
    EvaluationQuestion,
    ExpectedRelationship,
    load_evaluation_dataset,
)
from app.evaluation.metrics import (
    calculate_retrieval_metrics,
    calculate_graph_metrics,
    calculate_generation_metrics,
    calculate_citation_metrics,
)
from app.evaluation.runner import EvaluationRunner
from app.evaluation.reporting import EvaluationReporter

__all__ = [
    "EvaluationDataset",
    "EvaluationQuestion",
    "ExpectedRelationship",
    "load_evaluation_dataset",
    "calculate_retrieval_metrics",
    "calculate_graph_metrics",
    "calculate_generation_metrics",
    "calculate_citation_metrics",
    "EvaluationRunner",
    "EvaluationReporter",
]
