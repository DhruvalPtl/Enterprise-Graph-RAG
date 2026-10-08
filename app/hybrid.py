"""
Hybrid Retrieval Coordinator for Enterprise RAG.

This module coordinates:
1. Semantic dense vector retrieval via PostgreSQL + pgvector (HNSW index, MiniLM 384-d).
2. Lexical sparse keyword retrieval via Okapi BM25.
3. Rank fusion via Reciprocal Rank Fusion (RRF).

Architecture:
    User Query
        ├──> VectorStore.retrieve()   ──> [Vector Candidates] ──┐
        └──> BM25Retriever.search()   ──> [BM25 Candidates]   ──┴──> RRF ──> Top-K Results
"""
from typing import List, Dict, Any, Optional
import psycopg

from app.config import (
    VECTOR_TOP_K,
    BM25_TOP_K,
    RRF_TOP_K,
    RRF_K,
)
from app.models import AccessContext
from app.vector_store import VectorStore
from app.bm25 import BM25Retriever
from app.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """
    Coordinates multi-modal candidate retrieval (Dense Vector + Sparse BM25)
    and fuses them using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = RRF_K,
        vector_top_k: int = VECTOR_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        rrf_top_k: int = RRF_TOP_K,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_top_k = rrf_top_k

    def _ensure_bm25_retriever(self, conn: Optional[psycopg.Connection] = None) -> BM25Retriever:
        """Lazily initializes BM25 retriever from database or fallback JSON file if not provided."""
        if self.bm25_retriever is None:
            try:
                self.bm25_retriever = BM25Retriever.from_db(conn=conn)
            except Exception:
                self.bm25_retriever = BM25Retriever.from_file()
        return self.bm25_retriever

    def retrieve_with_details(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval and returns both individual system results and fused results,
        applying pre-retrieval access control filtering to both systems.

        Returns:
            {
                "query": query_text,
                "vector_results": List[Dict[str, Any]],
                "bm25_results": List[Dict[str, Any]],
                "fused_results": List[Dict[str, Any]],
            }
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty.")

        v_k = vector_top_k or self.vector_top_k
        b_k = bm25_top_k or self.bm25_top_k
        f_k = top_k or self.rrf_top_k
        fusion_k = rrf_k or self.rrf_k

        # 1. Execute dense vector search with access filtering
        vector_results = self.vector_store.retrieve(
            query_text=query_text,
            top_k=v_k,
            access_context=access_context,
            conn=conn,
        )

        # 2. Execute sparse BM25 search with access filtering
        bm25 = self._ensure_bm25_retriever(conn=conn)
        bm25_results = bm25.search(
            query_text=query_text,
            top_k=b_k,
            access_context=access_context,
        )

        # 3. Fuse ranked lists using RRF
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            k=fusion_k,
            top_k=f_k,
        )

        return {
            "query": query_text,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "fused_results": fused_results,
        }

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main entrypoint: executes hybrid retrieval and returns the top-k fused candidates.
        Unauthorized chunks are excluded before candidate generation.
        """
        details = self.retrieve_with_details(
            query_text=query_text,
            top_k=top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
            access_context=access_context,
            conn=conn,
        )
        return details["fused_results"]


def hybrid_retrieve(
    query_text: str,
    top_k: int = RRF_TOP_K,
    vector_top_k: int = VECTOR_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
    access_context: Optional[AccessContext] = None,
    conn: Optional[psycopg.Connection] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
) -> List[Dict[str, Any]]:
    """
    Functional convenience helper for hybrid retrieval.
    """
    retriever = hybrid_retriever or HybridRetriever(
        rrf_k=rrf_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        rrf_top_k=top_k,
    )
    return retriever.retrieve(
        query_text=query_text,
        top_k=top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
        rrf_k=rrf_k,
        access_context=access_context,
        conn=conn,
    )
