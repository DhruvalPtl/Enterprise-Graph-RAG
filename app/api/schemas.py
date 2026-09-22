"""
Pydantic Request and Response Schemas for the Enterprise RAG REST API (Step 7).

Ensures strict input validation (bounds, non-empty, whitespace stripping, forbidden extras)
and standardizes authoritative response structures.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.models import Citation, RAGResponse


class AccessContextSchema(BaseModel):
    """
    Caller authorization parameters for access-aware retrieval (Step 9).
    Default: strictly public access level and public department.
    """
    model_config = ConfigDict(extra="forbid")

    department: str = Field(
        default="public",
        description="Department of the caller (e.g. 'public', 'engineering', 'finance', 'hr').",
    )
    access_level: str = Field(
        default="public",
        description="Access clearance level: 'public', 'employee', 'manager', 'admin'.",
    )
    include_archived: bool = Field(
        default=False,
        description="Whether to include archived documents (default: False).",
    )

    @field_validator("access_level")
    @classmethod
    def validate_access_level(cls, v: str) -> str:
        cleaned = v.strip().lower()
        allowed = ["public", "employee", "manager", "admin"]
        if cleaned not in allowed:
            raise ValueError(f"Invalid access_level '{cleaned}'. Must be one of: {allowed}")
        return cleaned

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        return v.strip().lower()


class QueryRequest(BaseModel):
    """
    Input schema for the POST /query endpoint.
    Strictly validates user questions, bounds retrieval limits, and enforces caller access constraints.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "What is chunking and chunk overlap strategy?",
                "top_k": 5,
                "access_context": {
                    "department": "engineering",
                    "access_level": "employee",
                },
            }
        },
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search phrase or natural language question to ask the RAG platform.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of final reranked evidence passages to return (1 to 50).",
    )
    candidate_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Optional limit for Stage 1 hybrid retrieval candidate pool (1 to 100).",
    )
    access_context: Optional[AccessContextSchema] = Field(
        default=None,
        description="Optional caller access authorization context. Defaults strictly to public unprivileged access.",
    )
    retrieval_mode: Optional[str] = Field(
        default="hybrid_graph_vector",
        description="Retrieval engine mode: 'hybrid_graph_vector' or 'vector_bm25_rrf'.",
    )

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, v: Optional[str]) -> str:
        if v is None:
            return "hybrid_graph_vector"
        mode = v.strip().lower()
        if mode not in ("hybrid_graph_vector", "vector_bm25_rrf"):
            raise ValueError(f"Invalid retrieval_mode '{mode}'. Allowed: 'hybrid_graph_vector', 'vector_bm25_rrf'.")
        return mode

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query cannot be empty or contain only whitespace.")
        return cleaned


class CitationResponse(BaseModel):
    """
    Authoritative citation representation exposed via the REST API.
    Derived strictly from retrieved SearchResult database metadata.
    """
    source_id: int = Field(..., description="1-based citation index matching [SOURCE X] tags in the answer.")
    chunk_id: Any = Field(..., description="Unique database chunk identifier.")
    document_id: Any = Field(..., description="Parent document identifier.")
    filename: str = Field(..., description="Name of the source document.")
    page_number: Optional[int] = Field(None, description="1-based source page number if applicable.")
    section: Optional[str] = Field(None, description="Document section or header.")
    reranker_score: Optional[float] = Field(None, description="Fine Cross-Encoder relevance logit.")
    rrf_score: Optional[float] = Field(None, description="Stage 1 Reciprocal Rank Fusion score.")
    formatted: Optional[str] = Field(None, description="Human-readable formatted citation string.")
    retrieved_by: Optional[List[str]] = Field(None, description="Retrieval source provenance: 'vector', 'graph', or both.")
    content: Optional[str] = Field(None, description="Passage text content if available.")

    @classmethod
    def from_citation(cls, c: Citation) -> "CitationResponse":
        return cls(
            source_id=c.source_id,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            page_number=c.page_number,
            section=c.section,
            reranker_score=c.reranker_score,
            rrf_score=c.rrf_score,
            formatted=c.format_citation(),
            retrieved_by=c.sources.get("retrieved_by") if isinstance(c.sources, dict) else None,
            content=c.sources.get("content") if isinstance(c.sources, dict) else None,
        )


class QueryResponse(BaseModel):
    """
    Standardized response model for single-turn RAG question-answering.
    Contains grounded answer, authoritative citations, and operational diagnostics.
    """
    query: str = Field(..., description="The original normalized question.")
    answer: str = Field(..., description="Grounded answer produced by the LLM (or refusal if no evidence).")
    citations: List[CitationResponse] = Field(
        default_factory=list,
        description="Authoritative source citations corresponding to the evidence provided.",
    )
    model_name: Optional[str] = Field(None, description="The LLM model used for generation.")
    diagnostics: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Operational metrics (candidate counts, evidence counts, context budget).",
    )

    @classmethod
    def from_rag_response(cls, res: RAGResponse) -> "QueryResponse":
        return cls(
            query=res.query,
            answer=res.answer,
            citations=[CitationResponse.from_citation(c) for c in res.citations],
            model_name=res.model_name,
            diagnostics=res.diagnostics,
        )


class HealthResponse(BaseModel):
    """
    Liveness and readiness status model for the GET /health endpoint.
    """
    status: str = Field("ok", description="Overall health status: 'ok' or 'degraded'.")
    app: str = Field(..., description="Application title.")
    version: str = Field(..., description="Application version.")
    database: str = Field(..., description="PostgreSQL connection state: 'connected' or 'disconnected'.")
