"""
Graph + Vector Hybrid Retrieval Engine (Phase G4).

Orchestrates dual-path enterprise retrieval:
1. Path A (Vector + BM25 + RRF): High-recall semantic and lexical retrieval.
2. Path B (GraphRetriever): Entity linking and bounded graph traversal preserving provenance.
3. Evidence Fusion & Deduplication: Merges candidates by chunk_id, preserving provenance attribution
   (retrieved_by = ["vector"], ["graph"], or ["vector", "graph"]).
4. Cross-Encoder Reranking: Scores all fused candidates uniformly against the user query.
5. Strict Multi-Tenant Access Control: Multi-level authorization enforced at retrieval branches
   and verified at the fusion boundary.
"""
from typing import List, Dict, Any, Optional, Set, Union
import time
import psycopg

from app.config import (
    RERANKER_CANDIDATE_K,
    RERANKER_TOP_K,
)
from app.models import (
    AccessContext,
    SearchResult,
    GraphRetrievalResult,
    HybridRetrievalResult,
)
from app.reranker import RerankedRetrievalPipeline, CrossEncoderReranker
from app.graph_retriever import GraphRetriever


class GraphVectorHybridRetriever:
    """
    Coordinates Graph RAG and Vector/BM25 RAG, executing dual-path retrieval,
    deduplication, provenance attribution, and unified cross-encoder reranking.
    """

    def __init__(
        self,
        reranked_pipeline: Optional[RerankedRetrievalPipeline] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_k: int = RERANKER_CANDIDATE_K,
        top_k: int = RERANKER_TOP_K,
        graph_depth: int = 1,
        max_graph_results: int = 20,
        max_graph_seeds: int = 10,
    ):
        self.reranked_pipeline = reranked_pipeline or RerankedRetrievalPipeline(
            candidate_k=candidate_k,
            top_k=top_k,
        )
        self.graph_retriever = graph_retriever or GraphRetriever(
            default_depth=graph_depth,
            default_max_results=max_graph_results,
            default_max_seeds=max_graph_seeds,
        )
        # Reuse existing cross-encoder reranker instance to avoid redundant model loading
        self.reranker = reranker or self.reranked_pipeline.reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.graph_depth = graph_depth
        self.max_graph_results = max_graph_results
        self.max_graph_seeds = max_graph_seeds

    def retrieve_fused(
        self,
        query: str,
        candidate_k: Optional[int] = None,
        top_k: Optional[int] = None,
        access_context: Optional[AccessContext] = None,
        graph_depth: Optional[int] = None,
        max_graph_results: Optional[int] = None,
        conn: Optional[psycopg.Connection] = None,
    ) -> HybridRetrievalResult:
        """
        Executes dual-path retrieval, fuses evidence chunks, and reranks candidates.

        Args:
            query: User search query or question.
            candidate_k: Number of candidates for Vector+BM25 branch (default: 20).
            top_k: Final number of reranked evidence results (default: 5).
            access_context: Caller authorization context.
            graph_depth: Traversal depth for Graph branch (1 or 2).
            max_graph_results: Max relationships for Graph branch.
            conn: Optional PostgreSQL connection.

        Returns:
            HybridRetrievalResult with fused candidates, reranked results, and provenance.
        """
        start_total = time.perf_counter()
        clean_query = query.strip() if query else ""
        ctx = access_context or AccessContext()
        c_k = candidate_k or self.candidate_k
        t_k = top_k or self.top_k
        g_depth = graph_depth or self.graph_depth
        max_g_results = max_graph_results or self.max_graph_results

        # ----------------------------------------------------------------------
        # Empty query handling
        # ----------------------------------------------------------------------
        if not clean_query:
            return HybridRetrievalResult(
                query=query,
                reranked_results=[],
                graph_result=None,
                fused_candidates=[],
                vector_candidates=[],
                graph_candidates=[],
                deduplication_stats={
                    "vector_candidates": 0,
                    "graph_candidates": 0,
                    "overlap_count": 0,
                    "fused_count": 0,
                },
                access_context=ctx,
                execution_timing_ms={"total_ms": 0.0},
                retrieval_metadata={"status": "empty_query", "retrieval_mode": "hybrid_graph_vector"},
            )

        timing: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # Branch A: Existing Vector + BM25 + RRF Candidate Retrieval
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            hybrid_details = self.reranked_pipeline.hybrid_retriever.retrieve_with_details(
                query_text=clean_query,
                top_k=c_k,
                access_context=ctx,
                conn=conn,
            )
            raw_vector_candidates = hybrid_details.get("fused_results", [])
        except Exception as exc:
            raw_vector_candidates = []
        timing["vector_bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Normalize vector candidates into SearchResult objects
        vector_candidates: List[SearchResult] = []
        for cand in raw_vector_candidates:
            if isinstance(cand, SearchResult):
                sr = cand
            elif isinstance(cand, dict):
                sr = SearchResult(
                    chunk_id=cand.get("chunk_id"),
                    document_id=cand.get("document_id"),
                    content=cand.get("content", ""),
                    metadata=dict(cand.get("metadata", {})),
                    score=float(cand.get("score", 0.0)),
                    rank=int(cand.get("rank", 1)),
                    source="vector",
                    distance=cand.get("distance"),
                    rrf_score=cand.get("rrf_score"),
                    reranker_score=cand.get("reranker_score"),
                    sources=dict(cand.get("sources", {})),
                )
            else:
                continue

            # Ensure sources tracks provenance attribution
            if "retrieved_by" not in sr.sources:
                sr.sources["retrieved_by"] = ["vector"]
            elif "vector" not in sr.sources["retrieved_by"]:
                sr.sources["retrieved_by"].append("vector")
            vector_candidates.append(sr)

        # ----------------------------------------------------------------------
        # Branch B: Graph Knowledge Retrieval (Phase G3)
        # ----------------------------------------------------------------------
        t1 = time.perf_counter()
        try:
            graph_result: Optional[GraphRetrievalResult] = self.graph_retriever.retrieve(
                query=clean_query,
                access_context=ctx,
                depth=g_depth,
                max_results=max_g_results,
                max_seed_entities=self.max_graph_seeds,
                include_chunk_content=True,
                conn=conn,
            )
        except Exception as exc:
            graph_result = None
        timing["graph_retrieval_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        # Map graph provenance chunks to SearchResult candidates
        graph_candidates: List[SearchResult] = []
        if graph_result and graph_result.source_chunks:
            # Group relationships by chunk_id
            chunk_rel_map: Dict[int, List[Dict[str, Any]]] = {}
            for rel in graph_result.relationships:
                if rel.chunk_id is not None:
                    chunk_rel_map.setdefault(rel.chunk_id, []).append({
                        "source": rel.metadata.get("source_name", f"Entity_{rel.source_entity_id}"),
                        "relationship_type": rel.relationship_type,
                        "target": rel.metadata.get("target_name", f"Entity_{rel.target_entity_id}"),
                        "confidence": rel.metadata.get("confidence", 1.0),
                        "evidence": rel.metadata.get("evidence"),
                    })

            for rank_idx, chunk in enumerate(graph_result.source_chunks, start=1):
                cid = chunk.get("chunk_id")
                rels_for_chunk = chunk_rel_map.get(cid, [])

                g_cand = SearchResult(
                    chunk_id=cid,
                    document_id=chunk.get("document_id"),
                    content=chunk.get("content", ""),
                    metadata={
                        "document_name": chunk.get("document_name"),
                        "document_type": chunk.get("document_type"),
                        "page_number": chunk.get("page_number"),
                        "section": chunk.get("section", "General"),
                        "chunk_index": chunk.get("chunk_index"),
                        "department": chunk.get("department", "public"),
                        "access_level": chunk.get("access_level", "public"),
                        "status": chunk.get("status", "active"),
                        "graph_relationships": rels_for_chunk,
                    },
                    score=1.0,
                    rank=rank_idx,
                    source="graph",
                    sources={
                        "retrieved_by": ["graph"],
                        "graph": {
                            "relationships": rels_for_chunk,
                            "seed_entities": [e.canonical_name for e in graph_result.matched_entities],
                        },
                    },
                )
                graph_candidates.append(g_cand)

        # ----------------------------------------------------------------------
        # Phase C: Evidence Fusion & Deduplication (G4.3)
        # ----------------------------------------------------------------------
        t2 = time.perf_counter()
        fused_map: Dict[Any, SearchResult] = {}
        overlap_count = 0

        # Add vector candidates
        for vc in vector_candidates:
            fused_map[vc.chunk_id] = vc

        # Merge graph candidates
        for gc in graph_candidates:
            cid = gc.chunk_id
            if cid in fused_map:
                overlap_count += 1
                existing = fused_map[cid]
                # Combine provenance attribution
                retrieved_by: List[str] = existing.sources.get("retrieved_by", ["vector"])
                if "graph" not in retrieved_by:
                    retrieved_by.append("graph")
                existing.sources["retrieved_by"] = retrieved_by
                existing.sources["graph"] = gc.sources.get("graph", {})
                existing.source = "hybrid_graph_vector"

                # Attach graph relationships to metadata if not present
                existing_rels = existing.metadata.get("graph_relationships", [])
                new_rels = gc.metadata.get("graph_relationships", [])
                if new_rels and not existing_rels:
                    existing.metadata["graph_relationships"] = new_rels
            else:
                fused_map[cid] = gc

        # Phase D: Defense-in-Depth Access Control Validation (G4.9)
        authorized_fused: List[SearchResult] = []
        for cand in fused_map.values():
            meta = cand.metadata or {}
            dept = meta.get("department", "public")
            lvl = meta.get("access_level", "public")
            stat = meta.get("status", "active")
            if ctx.is_authorized(doc_department=dept, doc_access_level=lvl, doc_status=stat):
                authorized_fused.append(cand)

        timing["fusion_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        dedup_stats = {
            "vector_candidates": len(vector_candidates),
            "graph_candidates": len(graph_candidates),
            "overlap_count": overlap_count,
            "fused_count": len(authorized_fused),
        }

        # ----------------------------------------------------------------------
        # Phase E: Unified Cross-Encoder Reranking (G4.6)
        # ----------------------------------------------------------------------
        t3 = time.perf_counter()
        if authorized_fused:
            reranked_results = self.reranker.rerank(
                query_text=clean_query,
                candidates=authorized_fused,
                top_k=t_k,
            )
        else:
            reranked_results = []
        timing["reranker_ms"] = round((time.perf_counter() - t3) * 1000, 2)

        timing["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)

        retrieval_metadata = {
            "query": clean_query,
            "retrieval_mode": "hybrid_graph_vector",
            "candidate_k": c_k,
            "top_k": t_k,
            "graph_depth": g_depth,
            "graph_seed_count": len(graph_result.matched_entities) if graph_result else 0,
            "graph_relationship_count": len(graph_result.relationships) if graph_result else 0,
            "graph_connected_count": len(graph_result.connected_entities) if graph_result else 0,
            "reranked_count": len(reranked_results),
            "status": "success" if reranked_results else "no_evidence",
        }

        return HybridRetrievalResult(
            query=clean_query,
            reranked_results=reranked_results,
            graph_result=graph_result,
            fused_candidates=authorized_fused,
            vector_candidates=vector_candidates,
            graph_candidates=graph_candidates,
            deduplication_stats=dedup_stats,
            access_context=ctx,
            execution_timing_ms=timing,
            retrieval_metadata=retrieval_metadata,
        )


# Convenience alias for architectural clarity
HybridRAGRetriever = GraphVectorHybridRetriever
