"""
Dataset Loader and Schemas for Graph RAG Evaluation (Phase G5).
Defines Pydantic models for evaluation questions, expected entities, relationships,
and corpus provenance.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator


DEFAULT_EVALUATION_DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "graph_rag_eval.json"
)

VALID_CATEGORIES = {
    "direct_factual",
    "entity_relationship",
    "multi_hop_relationship",
    "comparison",
    "multiple_source_chunks",
    "vector_favored",
    "graph_favored",
    "no_evidence",
}


class ExpectedRelationship(BaseModel):
    """Represents an expected ground-truth relationship assertion."""
    source: str
    type: str
    target: str

    def tuple_key(self) -> tuple:
        return (self.source.strip().lower(), self.type.strip().upper(), self.target.strip().lower())


class ExpectedPage(BaseModel):
    """Represents a document and page reference for provenance verification."""
    document_id: int
    page_number: int


class EvaluationQuestion(BaseModel):
    """A single evaluation benchmark item."""
    id: str
    question: str
    category: str
    expected_answer: str
    expected_entities: List[str] = Field(default_factory=list)
    expected_relationships: List[ExpectedRelationship] = Field(default_factory=list)
    expected_source_chunks: List[int] = Field(default_factory=list)
    expected_pages: List[ExpectedPage] = Field(default_factory=list)
    notes: Optional[str] = ""

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{v}'. Must be one of: {sorted(list(VALID_CATEGORIES))}")
        return norm

    @property
    def is_no_evidence(self) -> bool:
        return self.category == "no_evidence" or len(self.expected_source_chunks) == 0


class EvaluationDataset(BaseModel):
    """Complete collection of benchmark evaluation questions."""
    dataset_version: str = "1.0"
    description: str = ""
    corpus_documents: List[Dict[str, Any]] = Field(default_factory=list)
    questions: List[EvaluationQuestion] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def get_by_id(self, question_id: str) -> Optional[EvaluationQuestion]:
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def filter_by_category(self, category: str) -> List[EvaluationQuestion]:
        cat_norm = category.strip().lower()
        return [q for q in self.questions if q.category == cat_norm]

    @property
    def categories(self) -> Set[str]:
        return {q.category for q in self.questions}

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for q in self.questions:
            counts[q.category] = counts.get(q.category, 0) + 1
        return counts


def load_evaluation_dataset(path: Optional[Union[str, Path]] = None) -> EvaluationDataset:
    """Loads and validates the benchmark evaluation dataset from a JSON file."""
    target_path = Path(path) if path else DEFAULT_EVALUATION_DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EvaluationDataset.model_validate(data)
