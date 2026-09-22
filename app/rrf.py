"""
Reciprocal Rank Fusion (RRF) Implementation for Enterprise RAG (Step 4).

RRF is a robust, parameter-insensitive rank fusion algorithm that merges ranked lists
from multiple independent retrieval systems (such as Dense Vector Search and Sparse BM25).

Mathematical Formula:
    RRF_score(d) = Σ [ 1 / (k + rank_m(d)) ] for each retriever m where d was retrieved

Key Properties:
- Operates strictly on ordinal rank positions (1, 2, 3...) rather than arbitrary score magnitudes.
- Avoids score calibration and scale mismatch (e.g. cosine similarity [0, 1] vs BM25 [0, ∞)).
- Grants substantial reinforcement to documents appearing near the top of multiple systems.
- Documents retrieved by only a single system remain eligible for the final candidate pool.
"""
from typing import List, Dict, Any, Optional
from app.config import RRF_K


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = RRF_K,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fuses multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked lists, where each list contains chunk result dictionaries.
                      Each chunk dictionary must contain at least 'chunk_id'.
        k: Smoothing constant preventing high ranks from dominating (default: 60).
        top_k: Optional maximum number of top fused results to return.

    Returns:
        A unified list of chunk dictionaries sorted by descending RRF score, with
        preserved content, metadata, fused rank, and individual source provenance.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be positive, got {k}")

    if not ranked_lists:
        return []

    # Map chunk_id -> merged accumulator record
    fused_scores: Dict[Any, float] = {}
    fused_records: Dict[Any, Dict[str, Any]] = {}
    fused_sources: Dict[Any, Dict[str, Dict[str, Any]]] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue

        for rank_idx, item in enumerate(ranked_list, start=1):
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue

            # Determine rank position: prioritize item's explicit 'rank' if given, else 1-based index
            item_rank = item.get("rank", rank_idx)
            source_name = item.get("source", "unknown")
            item_score = item.get("score")

            # RRF contribution formula: 1 / (k + rank)
            contribution = 1.0 / (k + item_rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution

            # Initialize record storage if seeing this chunk for the first time
            if chunk_id not in fused_records:
                fused_records[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": item.get("document_id"),
                    "content": item.get("content", ""),
                    "metadata": dict(item.get("metadata", {})),
                }
                fused_sources[chunk_id] = {}

            # Record provenance for this retrieval system
            fused_sources[chunk_id][source_name] = {
                "rank": item_rank,
                "score": item_score,
            }

    if not fused_scores:
        return []

    # Sort all candidates by descending RRF score
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda cid: fused_scores[cid],
        reverse=True,
    )

    if top_k is not None and top_k > 0:
        sorted_chunk_ids = sorted_chunk_ids[:top_k]

    results: List[Dict[str, Any]] = []
    for final_rank, cid in enumerate(sorted_chunk_ids, start=1):
        record = fused_records[cid]
        rrf_score = round(fused_scores[cid], 6)

        results.append({
            "chunk_id": record["chunk_id"],
            "document_id": record["document_id"],
            "content": record["content"],
            "score": rrf_score,
            "rrf_score": rrf_score,
            "rank": final_rank,
            "source": "hybrid",
            "sources": fused_sources[cid],
            "metadata": record["metadata"],
        })

    return results
