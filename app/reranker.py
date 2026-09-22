"""
Cross-Encoder Reranker and Two-Stage Retrieval Pipeline for Enterprise RAG (Step 5).

This module implements:
1. CrossEncoderReranker: Scores joint (query, chunk) candidate pairs using a pretrained
   cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) for deep cross-attention.
2. RerankedRetrievalPipeline: Coordinates Stage 1 (Hybrid RRF candidate generation) and
   Stage 2 (Cross-Encoder reranking) into a unified enterprise retrieval workflow.

Architecture:
    User Query
        │
    [Stage 1: High-Recall Candidate Retrieval]
        ├─► VectorStore (pgvector HNSW) ──► Top-K Vector Candidates ──┐
        └─► BM25Retriever (Sparse BM25) ──► Top-K BM25 Candidates   ──┴──► RRF (e.g. 20 candidates)
        │
    [Stage 2: High-Precision Reranking]
        └─► CrossEncoderReranker (ms-marco-MiniLM-L-6-v2)
                Pairs: [(query, chunk_1), (query, chunk_2), ...]
                Deep Full Cross-Attention Scoring
                Sort by descending logit score
                ──► Final Top-K Results (e.g. 5 results)
"""
from typing import List, Dict, Any, Optional, Union
import psycopg
from sentence_transformers import CrossEncoder

from app.config import (
    RERANKER_MODEL,
    RERANKER_CANDIDATE_K,
    RERANKER_TOP_K,
)
from app.models import SearchResult, AccessContext
from app.hybrid import HybridRetriever


class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder model.

    Unlike bi-encoders which encode query and document independently into vectors,
    a cross-encoder feeds both query and document text simultaneously into the
    transformer model. All cross-attention heads evaluate token-to-token interactions
    between the query and candidate chunk, producing high-precision relevance scores.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the CrossEncoder reranker.

        Args:
            model_name: HuggingFace model identifier (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
            model: Optional pre-initialized model instance or mock (useful for testing).
            batch_size: Batch size for model inference.
        """
        self.model_name = model_name or RERANKER_MODEL
        self.batch_size = batch_size

        if model is not None:
            self.model = model
        else:
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query_text: str,
        candidates: List[Union[SearchResult, Dict[str, Any]]],
        top_k: int = RERANKER_TOP_K,
    ) -> List[SearchResult]:
        """
        Scores candidate chunks against the query and returns the top_k reranked results.

        Args:
            query_text: User question or search query.
            candidates: Candidate chunks from Stage 1 retrieval (SearchResult or dicts).
            top_k: Number of reranked results to return.

        Returns:
            List of SearchResult objects sorted by descending reranker_score.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty.")

        if not candidates or top_k <= 0:
            return []

        # Standardize candidates into SearchResult objects
        normalized_candidates: List[SearchResult] = []
        for cand in candidates:
            if isinstance(cand, SearchResult):
                # Clone candidate to prevent mutating input objects in-place
                copied = SearchResult(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    content=cand.content,
                    metadata=dict(cand.metadata),
                    score=cand.score,
                    rank=cand.rank,
                    source=cand.source,
                    distance=cand.distance,
                    rrf_score=cand.rrf_score,
                    reranker_score=cand.reranker_score,
                    sources=dict(cand.sources),
                )
                normalized_candidates.append(copied)
            elif isinstance(cand, dict):
                normalized_candidates.append(
                    SearchResult(
                        chunk_id=cand.get("chunk_id"),
                        document_id=cand.get("document_id"),
                        content=cand.get("content", ""),
                        metadata=dict(cand.get("metadata", {})),
                        score=float(cand.get("score", 0.0)),
                        rank=int(cand.get("rank", 1)),
                        source=str(cand.get("source", "retrieval")),
                        distance=cand.get("distance"),
                        rrf_score=cand.get("rrf_score"),
                        reranker_score=cand.get("reranker_score"),
                        sources=dict(cand.get("sources", {})),
                    )
                )
            else:
                raise TypeError(f"Unsupported candidate type: {type(cand)}")

        # Construct pairs for cross-encoder inference: [[query, text_1], [query, text_2], ...]
        pairs = [[query_text, cand.content or ""] for cand in normalized_candidates]

        # Predict relevance logits/scores
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Assign reranker scores
        for idx, cand in enumerate(normalized_candidates):
            score_val = round(float(raw_scores[idx]), 6)
            cand.reranker_score = score_val
            cand.score = score_val
            cand.source = "reranked"
            # Record reranker rank and score in sources provenance dictionary
            cand.sources["reranker"] = {
                "score": score_val,
            }

        # Sort candidates strictly by descending reranker score
        reranked = sorted(
            normalized_candidates,
            key=lambda c: c.reranker_score if c.reranker_score is not None else float("-inf"),
            reverse=True,
        )

        # Slice to top_k
        top_reranked = reranked[:top_k]

        # Re-assign final 1-indexed rank
        for rank_idx, item in enumerate(top_reranked, start=1):
            item.rank = rank_idx
            item.sources["reranker"]["rank"] = rank_idx

        return top_reranked


class RerankedRetrievalPipeline:
    """
    Two-stage Enterprise Retrieval Pipeline:
    - Stage 1: Fast candidate generation via Hybrid Retriever (pgvector HNSW + BM25 + RRF)
    - Stage 2: High-precision reranking via CrossEncoder (ms-marco-MiniLM-L-6-v2)
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_k: int = RERANKER_CANDIDATE_K,
        top_k: int = RERANKER_TOP_K,
    ):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve_and_rerank(
        self,
        query_text: str,
        candidate_k: Optional[int] = None,
        top_k: Optional[int] = None,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> List[SearchResult]:
        """
        Executes Stage 1 candidate retrieval followed by Stage 2 cross-encoder reranking,
        enforcing pre-retrieval access control filtering.

        Args:
            query_text: Search query or question.
            candidate_k: Number of hybrid candidates to retrieve for reranking (default: 20).
            top_k: Final number of reranked results to return (default: 5).
            access_context: Optional caller authorization context.
            conn: Optional PostgreSQL connection.

        Returns:
            List of SearchResult objects sorted by descending reranker_score.
        """
        c_k = candidate_k or self.candidate_k
        t_k = top_k or self.top_k

        # Stage 1: Candidate generation (Hybrid RRF with access filtering)
        candidates = self.hybrid_retriever.retrieve(
            query_text=query_text,
            top_k=c_k,
            access_context=access_context,
            conn=conn,
        )

        # Stage 2: Cross-Encoder reranking
        return self.reranker.rerank(
            query_text=query_text,
            candidates=candidates,
            top_k=t_k,
        )

    def retrieve_with_diagnostics(
        self,
        query_text: str,
        candidate_k: Optional[int] = None,
        top_k: Optional[int] = None,
        access_context: Optional[AccessContext] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> Dict[str, Any]:
        """
        Executes both stages and returns comprehensive diagnostics:
        vector results, BM25 results, RRF candidates, and final reranked results.
        Useful for inspecting rank migrations and debugging retrieval accuracy.
        """
        c_k = candidate_k or self.candidate_k
        t_k = top_k or self.top_k

        hybrid_details = self.hybrid_retriever.retrieve_with_details(
            query_text=query_text,
            top_k=c_k,
            access_context=access_context,
            conn=conn,
        )

        candidates = hybrid_details["fused_results"]
        reranked = self.reranker.rerank(
            query_text=query_text,
            candidates=candidates,
            top_k=t_k,
        )

        return {
            "query": query_text,
            "vector_results": hybrid_details["vector_results"],
            "bm25_results": hybrid_details["bm25_results"],
            "candidates": candidates,
            "reranked_results": reranked,
        }


def rerank_results(
    query_text: str,
    candidates: List[Union[SearchResult, Dict[str, Any]]],
    top_k: int = RERANKER_TOP_K,
    reranker: Optional[CrossEncoderReranker] = None,
) -> List[SearchResult]:
    """
    Functional helper to rerank an existing candidate list.
    """
    r = reranker or CrossEncoderReranker()
    return r.rerank(query_text=query_text, candidates=candidates, top_k=top_k)


def retrieve_and_rerank(
    query_text: str,
    candidate_k: int = RERANKER_CANDIDATE_K,
    top_k: int = RERANKER_TOP_K,
    access_context: Optional[AccessContext] = None,
    pipeline: Optional[RerankedRetrievalPipeline] = None,
    conn: Optional[psycopg.Connection] = None,
) -> List[SearchResult]:
    """
    Functional helper to execute full two-stage retrieval and reranking.
    """
    p = pipeline or RerankedRetrievalPipeline(candidate_k=candidate_k, top_k=top_k)
    return p.retrieve_and_rerank(
        query_text=query_text,
        candidate_k=candidate_k,
        top_k=top_k,
        access_context=access_context,
        conn=conn,
    )
