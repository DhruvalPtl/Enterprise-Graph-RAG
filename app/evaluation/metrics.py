"""
Pure Deterministic Evaluation Metrics for Graph RAG (Phase G5).
Calculates retrieval, graph-specific, citation, and generation metrics
with zero external network or LLM dependencies for reproducibility.
"""
import re
from typing import Any, Dict, List, Optional, Set
from app.prompts import NO_EVIDENCE_MESSAGE


def calculate_recall_at_k(retrieved_chunk_ids: List[int], expected_chunk_ids: List[int], k: int = 5) -> float:
    """Calculates Recall@K: fraction of expected chunks retrieved in the top K."""
    if not expected_chunk_ids:
        # Negative control / no evidence case: perfect recall if zero retrieved or empty expected
        return 1.0 if not retrieved_chunk_ids else 0.0

    top_k_set = set(retrieved_chunk_ids[:k])
    gold_set = set(expected_chunk_ids)
    hits = len(top_k_set.intersection(gold_set))
    return float(hits / len(gold_set))


def calculate_hit_rate_at_k(retrieved_chunk_ids: List[int], expected_chunk_ids: List[int], k: int = 5) -> float:
    """Calculates HitRate@K: 1.0 if at least one expected chunk is in top K, else 0.0."""
    if not expected_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    top_k_set = set(retrieved_chunk_ids[:k])
    gold_set = set(expected_chunk_ids)
    return 1.0 if bool(top_k_set.intersection(gold_set)) else 0.0


def calculate_mrr(retrieved_chunk_ids: List[int], expected_chunk_ids: List[int]) -> float:
    """Calculates Reciprocal Rank (1/rank) of the first expected chunk found (1-indexed)."""
    if not expected_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    gold_set = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def calculate_retrieval_metrics(
    retrieved_chunk_ids: List[int],
    expected_chunk_ids: List[int],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Computes standard information retrieval metrics across multiple K cutoffs."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = calculate_recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k)
        metrics[f"hit_rate@{k}"] = calculate_hit_rate_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k)

    metrics["mrr"] = calculate_mrr(retrieved_chunk_ids, expected_chunk_ids)
    return metrics


def calculate_graph_metrics(
    vector_candidate_chunk_ids: List[int],
    graph_candidate_chunk_ids: List[int],
    expected_chunk_ids: List[int],
    matched_seed_count: int = 0,
    retrieved_relationships: Optional[List[Dict[str, Any]]] = None,
    expected_relationships: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Computes Graph-specific evaluation metrics:
    - seed hit rate
    - relationship retrieval rate
    - graph provenance coverage
    - graph/vector overlap rate (Jaccard index)
    - additional unique evidence contributed by graph
    - additional unique relevant evidence contributed by graph
    """
    vec_set = set(vector_candidate_chunk_ids)
    graph_set = set(graph_candidate_chunk_ids)
    gold_set = set(expected_chunk_ids)

    # 1. Seed Hit Rate
    seed_hit = 1.0 if matched_seed_count > 0 else 0.0

    # 2. Relationship retrieval rate
    rel_retrieval_rate = 0.0
    if expected_relationships:
        retrieved_edges = set()
        if retrieved_relationships:
            for r in retrieved_relationships:
                s = str(r.get("source_name") or r.get("source", "")).strip().lower()
                t = str(r.get("target_name") or r.get("target", "")).strip().lower()
                rtype = str(r.get("relationship_type", "")).strip().upper()
                retrieved_edges.add((s, rtype, t))
                retrieved_edges.add((t, rtype, s))  # bidirectional matching tolerance

        matched_rels = 0
        for exp in expected_relationships:
            exp_tuple = exp.tuple_key() if hasattr(exp, "tuple_key") else (
                str(exp.get("source", "")).strip().lower(),
                str(exp.get("type", "")).strip().upper(),
                str(exp.get("target", "")).strip().lower(),
            )
            # Check direct or reverse match
            if exp_tuple in retrieved_edges or (exp_tuple[2], exp_tuple[1], exp_tuple[0]) in retrieved_edges:
                matched_rels += 1
        rel_retrieval_rate = matched_rels / len(expected_relationships)

    # 3. Graph Provenance Coverage
    graph_prov_cov = 0.0
    if gold_set:
        graph_prov_cov = len(graph_set.intersection(gold_set)) / len(gold_set)
    elif not gold_set and not graph_set:
        graph_prov_cov = 1.0

    # 4. Graph / Vector Overlap (Jaccard)
    union_len = len(vec_set.union(graph_set))
    inter_len = len(vec_set.intersection(graph_set))
    overlap_rate = (inter_len / union_len) if union_len > 0 else 0.0

    # 5. Additional Unique Evidence Contributed by Graph
    unique_graph_chunks = graph_set - vec_set
    unique_graph_count = len(unique_graph_chunks)

    # 6. Additional Unique Relevant Evidence Contributed by Graph
    unique_relevant_graph_chunks = unique_graph_chunks.intersection(gold_set)
    unique_relevant_count = len(unique_relevant_graph_chunks)

    return {
        "graph_seed_hit": seed_hit,
        "relationship_retrieval_rate": rel_retrieval_rate,
        "graph_provenance_coverage": graph_prov_cov,
        "graph_vector_overlap_rate": overlap_rate,
        "graph_overlap_count": inter_len,
        "unique_graph_chunks_count": unique_graph_count,
        "unique_relevant_graph_chunks_count": unique_relevant_count,
        "unique_relevant_chunk_ids": sorted(list(unique_relevant_graph_chunks)),
    }


def calculate_citation_metrics(
    cited_chunk_ids: List[int],
    expected_chunk_ids: List[int],
) -> Dict[str, float]:
    """
    Computes Citation Correctness and Citation Completeness:
    - Citation Correctness: Fraction of citations that cite a true gold chunk.
    - Citation Completeness / Recall: Fraction of gold chunks cited in the answer.
    """
    if not expected_chunk_ids:
        # Negative control / no evidence query
        return {
            "citation_correctness": 1.0 if not cited_chunk_ids else 0.0,
            "citation_completeness": 1.0 if not cited_chunk_ids else 0.0,
        }

    if not cited_chunk_ids:
        return {
            "citation_correctness": 0.0,
            "citation_completeness": 0.0,
        }

    cited_set = set(cited_chunk_ids)
    gold_set = set(expected_chunk_ids)

    # Correctness: precision of citations
    correct_citations = len(cited_set.intersection(gold_set))
    correctness = correct_citations / len(cited_set)

    # Completeness: recall of citations
    completeness = correct_citations / len(gold_set)

    return {
        "citation_correctness": correctness,
        "citation_completeness": completeness,
    }


def calculate_generation_metrics(
    generated_answer: str,
    expected_answer: str,
    expected_entities: List[str],
    retrieved_chunk_contents: List[str],
    is_no_evidence_query: bool = False,
) -> Dict[str, float]:
    """
    Deterministic generation metrics without non-deterministic LLM judge calls:
    - entity_coverage: Fraction of expected key entities present in the answer.
    - refusal_correctness: Correct refusal detection on no-evidence queries.
    - groundedness_score: Fraction of answer key sentences supported by retrieved chunks.
    - answer_correctness: Blended deterministic factual score.
    """
    clean_ans = generated_answer.strip()

    # 1. No evidence check
    if is_no_evidence_query:
        refusal_phrases = [
            "not provide sufficient information",
            "insufficient information",
            "does not contain",
            "no evidence",
            "not mentioned",
            "unsupported",
        ]
        is_refusal = any(p in clean_ans.lower() for p in refusal_phrases)
        score = 1.0 if is_refusal else 0.0
        return {
            "entity_coverage": score,
            "refusal_correctness": score,
            "groundedness_score": score,
            "answer_correctness": score,
        }

    # 2. Entity Coverage in Answer
    entity_hits = 0
    if expected_entities:
        for ent in expected_entities:
            # Case-insensitive substring or word boundary match
            pattern = re.compile(rf"\b{re.escape(ent.lower())}\b", re.IGNORECASE)
            if pattern.search(clean_ans) or ent.lower() in clean_ans.lower():
                entity_hits += 1
        entity_coverage = entity_hits / len(expected_entities)
    else:
        entity_coverage = 1.0

    # 3. Groundedness Score (Lexical N-Gram / Sentence Containment in Context)
    combined_context = " ".join(retrieved_chunk_contents).lower()
    sentences = [s.strip() for s in re.split(r"[.!?\n]", clean_ans) if len(s.strip()) > 15]

    grounded_count = 0
    for sentence in sentences:
        words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
        if not words:
            continue
        # Check if majority of informative words in sentence appear in context
        in_context = sum(1 for w in words if w in combined_context)
        if (in_context / len(words)) >= 0.65:
            grounded_count += 1

    groundedness = (grounded_count / len(sentences)) if sentences else 1.0

    # 4. Overall Deterministic Answer Correctness
    answer_correctness = (entity_coverage * 0.7) + (groundedness * 0.3)

    return {
        "entity_coverage": entity_coverage,
        "refusal_correctness": 1.0,
        "groundedness_score": groundedness,
        "answer_correctness": answer_correctness,
    }
