"""
Vector Store and Similarity Retrieval Layer for Enterprise RAG.

This module implements vector similarity search using PostgreSQL with the pgvector extension.
It handles:
- Embedding incoming queries using the primary MiniLM model (384 dimensions).
- Strict dimension validation (rejecting non-384 vectors like 768-d Gemini vectors).
- Cosine distance query execution (<=>) leveraging the HNSW index on the chunks table.
- Formatted retrieval of top-k chunks with content, metadata, distance, and similarity score.
"""
from typing import List, Dict, Any, Optional
import psycopg
from psycopg.rows import dict_row

from app.config import VECTOR_DIMENSION
from app.db import get_connection
from app.embeddings import EmbeddingService
from app.models import AccessContext


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    access_context: Optional[AccessContext] = None,
    conn: Optional[psycopg.Connection] = None,
) -> List[Dict[str, Any]]:
    """
    Performs cosine vector similarity search against PostgreSQL chunks table with access-control filtering.

    Args:
        query_embedding: A 384-dimensional float vector.
        top_k: Maximum number of most relevant chunks to return (default: 5).
        access_context: Optional caller authorization context. If None, defaults strictly to public-only access.
        conn: Optional existing psycopg connection. If None, opens and closes one cleanly.

    Returns:
        A list of dictionaries ordered by ascending cosine distance (descending similarity).
        Unauthorized chunks are completely excluded in PostgreSQL.
    """
    if query_embedding is None:
        raise ValueError("query_embedding cannot be None.")

    emb_len = len(query_embedding)
    if emb_len != VECTOR_DIMENSION:
        raise ValueError(
            f"Incompatible query vector dimension: expected {VECTOR_DIMENSION} (MiniLM), "
            f"got {emb_len}. Cosine vector search in this database requires 384-d vectors."
        )

    if top_k <= 0:
        return []

    # Default to strictly unprivileged public access if not provided
    ctx = access_context or AccessContext()
    allowed_levels = ctx.allowed_access_levels()

    should_close = False
    if conn is None:
        conn = get_connection(autocommit=True)
        should_close = True

    try:
        from pgvector.psycopg import register_vector
        register_vector(conn)
    except Exception:
        pass

    # Cosine distance query with pre-retrieval authorization constraints
    sql = """
    SELECT
        c.id AS chunk_id,
        c.document_id,
        c.content,
        c.page_number,
        c.section,
        c.chunk_index,
        d.filename AS document_name,
        d.document_type,
        d.department,
        d.access_level,
        d.status,
        (c.embedding <=> %s::vector) AS distance
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
      AND (%s OR d.status = 'active')
      AND (%s OR d.access_level = ANY(%s))
      AND (%s OR d.department = 'public' OR d.department = %s)
    ORDER BY distance ASC
    LIMIT %s;
    """
    params = (
        query_embedding,
        ctx.include_archived,
        ctx.is_admin,
        allowed_levels,
        ctx.is_admin,
        ctx.department,
        top_k,
    )

    results: List[Dict[str, Any]] = []

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            for rank, row in enumerate(rows, start=1):
                dist = float(row["distance"])
                sim = max(0.0, 1.0 - dist)
                results.append({
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "distance": round(dist, 4),
                    "similarity": round(sim, 4),
                    "score": round(sim, 4),
                    "rank": rank,
                    "source": "vector",
                    "metadata": {
                        "document_name": row["document_name"],
                        "document_type": row["document_type"],
                        "page_number": row["page_number"],
                        "section": row["section"],
                        "chunk_index": row["chunk_index"],
                        "department": row.get("department", "public"),
                        "access_level": row.get("access_level", "public"),
                        "status": row.get("status", "active"),
                    },
                })
        return results

    finally:
        if should_close:
            conn.close()


def retrieve_similar_chunks(
    query_text: str,
    top_k: int = 5,
    access_context: Optional[AccessContext] = None,
    conn: Optional[psycopg.Connection] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> List[Dict[str, Any]]:
    """
    End-to-end vector retrieval:
    1. Embeds query_text using the primary MiniLM model (384-d).
    2. Searches PostgreSQL pgvector for top-k matching chunks via cosine distance,
       applying authorization filters before candidate selection.

    Args:
        query_text: The user question or search phrase.
        top_k: Number of chunks to retrieve.
        access_context: Optional caller access context.
        conn: Optional active database connection.
        embedding_service: Optional EmbeddingService instance (defaults to sentence-transformers).

    Returns:
        Ranked list of authorized chunk dictionaries with similarity scores and metadata.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty.")

    service = embedding_service or EmbeddingService(provider="sentence-transformers")
    query_embedding = service.embed_text(query_text)

    return search_similar_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        access_context=access_context,
        conn=conn,
    )


class VectorStore:
    """
    Object-oriented facade for vector similarity operations in Enterprise RAG.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService(provider="sentence-transformers")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """Searches for chunks similar to the provided embedding vector within authorized documents."""
        return search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            access_context=access_context,
            conn=conn,
        )

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """Embeds query_text and retrieves top-k authorized similar chunks."""
        return retrieve_similar_chunks(
            query_text=query_text,
            top_k=top_k,
            access_context=access_context,
            conn=conn,
            embedding_service=self.embedding_service,
        )
