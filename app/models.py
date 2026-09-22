"""
Data models for Documents and Chunks.

Using standard Python dataclasses keeps our foundation lightweight,
transparent, and dependency-free before introducing API frameworks later.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import hashlib


def generate_content_hash(text: str) -> str:
    """
    Generate a deterministic, short SHA-256 hash from text or file content.
    Useful for idempotent document and chunk IDs.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# Access Level Hierarchy: public (0) < employee (1) < manager (2) < admin (3)
ACCESS_LEVEL_HIERARCHY: Dict[str, int] = {
    "public": 0,
    "employee": 1,
    "manager": 2,
    "admin": 3,
}


@dataclass
class AccessContext:
    """
    Caller access authorization context for metadata-filtered retrieval (Step 9).
    Enforces that authorization is checked BEFORE candidates enter vector, BM25, or RRF stages.
    """
    department: str = "public"
    access_level: str = "public"
    include_archived: bool = False

    def __post_init__(self):
        self.department = (self.department or "public").strip().lower()
        self.access_level = (self.access_level or "public").strip().lower()
        if self.access_level not in ACCESS_LEVEL_HIERARCHY:
            raise ValueError(
                f"Invalid access_level '{self.access_level}'. "
                f"Allowed values are: {list(ACCESS_LEVEL_HIERARCHY.keys())}"
            )

    @property
    def is_admin(self) -> bool:
        return self.access_level == "admin"

    def allowed_access_levels(self) -> List[str]:
        """Returns the list of document access levels this context is permitted to view."""
        if self.is_admin:
            return list(ACCESS_LEVEL_HIERARCHY.keys())
        user_rank = ACCESS_LEVEL_HIERARCHY.get(self.access_level, 0)
        return [lvl for lvl, rank in ACCESS_LEVEL_HIERARCHY.items() if rank <= user_rank]

    def is_authorized(
        self,
        doc_department: Optional[str] = None,
        doc_access_level: Optional[str] = None,
        doc_status: Optional[str] = None,
    ) -> bool:
        """
        Determines whether a document/chunk is accessible to this context.
        Used for in-memory BM25 filtering, vector filtering validation, and unit tests.
        """
        doc_dept = (doc_department or "public").strip().lower()
        doc_level = (doc_access_level or "public").strip().lower()
        doc_stat = (doc_status or "active").strip().lower()

        # 1. Status check: archived documents are inaccessible unless explicitly permitted
        if doc_stat == "archived" and not self.include_archived:
            return False

        # 2. Admin has universal read access across all departments and levels
        if self.is_admin:
            return True

        # 3. Access level hierarchy check
        user_rank = ACCESS_LEVEL_HIERARCHY.get(self.access_level, 0)
        doc_rank = ACCESS_LEVEL_HIERARCHY.get(doc_level, 0)
        if user_rank < doc_rank:
            return False

        # 4. Department isolation check
        # Documents with department 'public' are readable by anyone whose access level qualifies
        if doc_dept == "public":
            return True

        # Department-specific documents require exact matching department
        return self.department == doc_dept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "department": self.department,
            "access_level": self.access_level,
            "include_archived": self.include_archived,
        }


@dataclass
class DocumentPage:
    """Represents a single page or segment from a document."""
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Represents an ingested document before chunking with access metadata (Step 9).
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[DocumentPage] = field(default_factory=list)
    department: str = "public"
    access_level: str = "public"
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Document instance to a plain dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "total_pages": len(self.pages) if self.pages else 1,
            "department": self.department,
            "access_level": self.access_level,
            "status": self.status,
        }


@dataclass
class Chunk:
    """
    Represents an atomic text chunk ready for downstream embedding and retrieval.
    """
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Chunk instance to dictionary matching the target enterprise schema."""
        data: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding
        if self.embedding_provider is not None:
            data["embedding_provider"] = self.embedding_provider
        if self.embedding_model is not None:
            data["embedding_model"] = self.embedding_model
        if self.embedding_dimension is not None:
            data["embedding_dimension"] = self.embedding_dimension
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Reconstruct a Chunk instance from dictionary representation."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            embedding_provider=data.get("embedding_provider"),
            embedding_model=data.get("embedding_model"),
            embedding_dimension=data.get("embedding_dimension"),
        )


@dataclass
class SearchResult:
    """
    Standardized result model for vector, BM25, and hybrid retrieval.
    Preserves chunk identification, document origin, full text, score, and source ranks.
    """
    chunk_id: Any
    document_id: Any
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 1
    source: str = "retrieval"
    distance: Optional[float] = None
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    sources: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts SearchResult to standard dictionary."""
        data: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
        }
        if self.distance is not None:
            data["distance"] = self.distance
        if self.rrf_score is not None:
            data["rrf_score"] = self.rrf_score
        if self.reranker_score is not None:
            data["reranker_score"] = self.reranker_score
        if self.sources:
            data["sources"] = self.sources
        return data


@dataclass
class Citation:
    """
    Authoritative citation representation constructed from retrieved SearchResult metadata.
    Maps to stable [SOURCE X] tags used in the LLM prompt.
    """
    source_id: int
    chunk_id: Any
    document_id: Any
    filename: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    reranker_score: Optional[float] = None
    rrf_score: Optional[float] = None
    sources: Dict[str, Any] = field(default_factory=dict)

    def format_citation(self) -> str:
        """Formats citation as a clean, human-readable reference line."""
        parts = [f"[{self.source_id}] {self.filename}"]
        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")
        if self.section and self.section != "General":
            parts.append(f"Section: {self.section}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Converts Citation to dictionary representation."""
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "section": self.section,
            "reranker_score": self.reranker_score,
            "rrf_score": self.rrf_score,
            "sources": self.sources,
        }


@dataclass
class RAGResponse:
    """
    Complete response model for single-turn enterprise RAG question-answering.
    Contains generated answer, authoritative citations, and retrieval diagnostics.
    """
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_results: List[SearchResult] = field(default_factory=list)
    model_name: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts RAGResponse to standardized dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "model_name": self.model_name,
            "retrieved_count": len(self.retrieved_results),
            "diagnostics": self.diagnostics,
        }


@dataclass
class Entity:
    """
    Represents a canonical semantic entity node in the Graph RAG data model (Phase G1).
    """
    id: Optional[int]
    canonical_name: str
    entity_type: str
    display_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "display_name": self.display_name,
            "metadata": dict(self.metadata),
            "created_at": str(self.created_at) if self.created_at else None,
            "updated_at": str(self.updated_at) if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data.get("id"),
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            display_name=data["display_name"],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class Relationship:
    """
    Represents a directed semantic relationship edge with source material provenance (Phase G1).
    """
    id: Optional[int]
    source_entity_id: int
    target_entity_id: int
    relationship_type: str
    document_id: int
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "metadata": dict(self.metadata),
            "created_at": str(self.created_at) if self.created_at else None,
            "updated_at": str(self.updated_at) if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            id=data.get("id"),
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            relationship_type=data["relationship_type"],
            document_id=data["document_id"],
            chunk_id=data.get("chunk_id"),
            page_number=data.get("page_number"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class GraphRetrievalResult:
    """
    Structured outcome of Graph RAG retrieval (Phase G3).
    Encapsulates matched seed entities, traversed semantic relationships,
    connected neighbor entities, and full provenance linkage (chunks, docs, pages).
    """
    matched_entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    connected_entities: List[Entity] = field(default_factory=list)
    source_chunk_ids: List[int] = field(default_factory=list)
    document_ids: List[int] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Returns True if no entities or relationships were retrieved."""
        return len(self.matched_entities) == 0 and len(self.relationships) == 0

    @property
    def total_entities(self) -> int:
        """Total distinct entities involved in this retrieval result."""
        return len(self.matched_entities) + len(self.connected_entities)

    def to_dict(self) -> Dict[str, Any]:
        """Converts GraphRetrievalResult to a JSON-serializable dictionary."""
        return {
            "matched_entities": [e.to_dict() for e in self.matched_entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "connected_entities": [e.to_dict() for e in self.connected_entities],
            "source_chunk_ids": list(self.source_chunk_ids),
            "document_ids": list(self.document_ids),
            "page_numbers": list(self.page_numbers),
            "retrieval_metadata": dict(self.retrieval_metadata),
            "source_chunks": list(self.source_chunks),
            "total_entities": self.total_entities,
            "is_empty": self.is_empty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRetrievalResult":
        """Deserializes a dictionary into a GraphRetrievalResult instance."""
        return cls(
            matched_entities=[Entity.from_dict(e) for e in data.get("matched_entities", [])],
            relationships=[Relationship.from_dict(r) for r in data.get("relationships", [])],
            connected_entities=[Entity.from_dict(e) for e in data.get("connected_entities", [])],
            source_chunk_ids=data.get("source_chunk_ids", []),
            document_ids=data.get("document_ids", []),
            page_numbers=data.get("page_numbers", []),
            retrieval_metadata=data.get("retrieval_metadata", {}),
            source_chunks=data.get("source_chunks", []),
        )


@dataclass
class HybridRetrievalResult:
    """
    Structured outcome of Graph + Vector Hybrid RAG retrieval (Phase G4).
    Preserves vector/BM25 candidates, graph traversal results, fused and deduplicated candidates,
    provenance attribution ('vector', 'graph', 'vector+graph'), cross-encoder reranked results,
    and operational diagnostics.
    """
    query: str
    reranked_results: List[SearchResult] = field(default_factory=list)
    graph_result: Optional[GraphRetrievalResult] = None
    fused_candidates: List[SearchResult] = field(default_factory=list)
    vector_candidates: List[SearchResult] = field(default_factory=list)
    graph_candidates: List[SearchResult] = field(default_factory=list)
    deduplication_stats: Dict[str, Any] = field(default_factory=dict)
    access_context: Optional[AccessContext] = None
    execution_timing_ms: Dict[str, float] = field(default_factory=dict)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Returns True if no reranked evidence or candidates were retrieved."""
        return len(self.reranked_results) == 0 and len(self.fused_candidates) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts HybridRetrievalResult to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "graph_result": self.graph_result.to_dict() if self.graph_result else None,
            "fused_candidates": [r.to_dict() for r in self.fused_candidates],
            "vector_candidates": [r.to_dict() for r in self.vector_candidates],
            "graph_candidates": [r.to_dict() for r in self.graph_candidates],
            "deduplication_stats": dict(self.deduplication_stats),
            "access_context": self.access_context.to_dict() if self.access_context else None,
            "execution_timing_ms": dict(self.execution_timing_ms),
            "retrieval_metadata": dict(self.retrieval_metadata),
            "is_empty": self.is_empty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridRetrievalResult":
        """Deserializes a dictionary into a HybridRetrievalResult instance."""
        graph_data = data.get("graph_result")
        ctx_data = data.get("access_context")

        def _to_sr(item: Any) -> SearchResult:
            if isinstance(item, SearchResult):
                return item
            if isinstance(item, dict):
                return SearchResult(
                    chunk_id=item.get("chunk_id"),
                    document_id=item.get("document_id"),
                    content=item.get("content", ""),
                    metadata=dict(item.get("metadata", {})),
                    score=float(item.get("score", 0.0)),
                    rank=int(item.get("rank", 1)),
                    source=str(item.get("source", "retrieval")),
                    distance=item.get("distance"),
                    rrf_score=item.get("rrf_score"),
                    reranker_score=item.get("reranker_score"),
                    sources=dict(item.get("sources", {})),
                )
            raise TypeError(f"Cannot parse SearchResult from {type(item)}")

        return cls(
            query=data.get("query", ""),
            reranked_results=[_to_sr(r) for r in data.get("reranked_results", [])],
            graph_result=GraphRetrievalResult.from_dict(graph_data) if graph_data else None,
            fused_candidates=[_to_sr(r) for r in data.get("fused_candidates", [])],
            vector_candidates=[_to_sr(r) for r in data.get("vector_candidates", [])],
            graph_candidates=[_to_sr(r) for r in data.get("graph_candidates", [])],
            deduplication_stats=dict(data.get("deduplication_stats", {})),
            access_context=AccessContext(**ctx_data) if ctx_data else None,
            execution_timing_ms=dict(data.get("execution_timing_ms", {})),
            retrieval_metadata=dict(data.get("retrieval_metadata", {})),
        )







