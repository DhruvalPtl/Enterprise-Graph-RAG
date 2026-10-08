"""
Unit Tests for Reciprocal Rank Fusion (RRF).

Verifies:
- Standard RRF formula: Score(d) = Σ 1 / (k + rank_m(d))
- Deterministic rank order calculation
- Multi-system reinforcement (chunks appearing in both lists receive higher scores)
- Single-system candidate eligibility (chunks appearing in only one list remain eligible)
- Configurable k smoothing parameter
- Preservation of chunk metadata and source attribution
- Edge cases (empty lists, negative k)
"""
import pytest
from app.rrf import reciprocal_rank_fusion


def test_rrf_deterministic_example():
    """
    Verifies exact math for the reference example:
    Vector: A(1), B(2), C(3)
    BM25:   C(1), A(2), D(3)
    k = 60

    Expected RRF scores:
    - A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0163934 + 0.0161290 = 0.0325225
    - C: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.0158730 + 0.0163934 = 0.0322665
    - B: 1/(60+2) = 1/62 ≈ 0.0161290
    - D: 1/(60+3) = 1/63 ≈ 0.0158730

    Expected ordering: A > C > B > D
    """
    vector_list = [
        {"chunk_id": "A", "rank": 1, "source": "vector", "score": 0.95, "content": "A content"},
        {"chunk_id": "B", "rank": 2, "source": "vector", "score": 0.85, "content": "B content"},
        {"chunk_id": "C", "rank": 3, "source": "vector", "score": 0.75, "content": "C content"},
    ]
    bm25_list = [
        {"chunk_id": "C", "rank": 1, "source": "bm25", "score": 4.5, "content": "C content"},
        {"chunk_id": "A", "rank": 2, "source": "bm25", "score": 3.2, "content": "A content"},
        {"chunk_id": "D", "rank": 3, "source": "bm25", "score": 1.8, "content": "D content"},
    ]

    fused = reciprocal_rank_fusion([vector_list, bm25_list], k=60)

    assert len(fused) == 4
    # Rank 1: A
    assert fused[0]["chunk_id"] == "A"
    assert fused[0]["rank"] == 1
    assert fused[0]["rrf_score"] == pytest.approx(0.032522, abs=1e-5)
    assert "vector" in fused[0]["sources"]
    assert "bm25" in fused[0]["sources"]
    assert fused[0]["sources"]["vector"]["rank"] == 1
    assert fused[0]["sources"]["bm25"]["rank"] == 2

    # Rank 2: C
    assert fused[1]["chunk_id"] == "C"
    assert fused[1]["rank"] == 2
    assert fused[1]["rrf_score"] == pytest.approx(0.032266, abs=1e-5)

    # Rank 3: B (only in vector)
    assert fused[2]["chunk_id"] == "B"
    assert fused[2]["rank"] == 3
    assert fused[2]["rrf_score"] == pytest.approx(0.016129, abs=1e-5)
    assert "vector" in fused[2]["sources"]
    assert "bm25" not in fused[2]["sources"]

    # Rank 4: D (only in BM25)
    assert fused[3]["chunk_id"] == "D"
    assert fused[3]["rank"] == 4
    assert fused[3]["rrf_score"] == pytest.approx(0.015873, abs=1e-5)
    assert "bm25" in fused[3]["sources"]
    assert "vector" not in fused[3]["sources"]


def test_rrf_configurable_k():
    """Verifies that changing k properly shifts relative score weights."""
    vec = [{"chunk_id": "A", "rank": 1, "source": "vector"}]
    bm = [{"chunk_id": "A", "rank": 1, "source": "bm25"}]

    # With k=20: 1/(20+1) + 1/(20+1) = 2/21 ≈ 0.095238
    fused_k20 = reciprocal_rank_fusion([vec, bm], k=20)
    assert fused_k20[0]["rrf_score"] == pytest.approx(2.0 / 21.0, abs=1e-5)

    # With k=100: 1/(100+1) + 1/(100+1) = 2/101 ≈ 0.019802
    fused_k100 = reciprocal_rank_fusion([vec, bm], k=100)
    assert fused_k100[0]["rrf_score"] == pytest.approx(2.0 / 101.0, abs=1e-5)


def test_rrf_invalid_k_raises_error():
    """Verifies ValueError when k <= 0."""
    with pytest.raises(ValueError, match="RRF constant k must be positive"):
        reciprocal_rank_fusion([], k=0)
    with pytest.raises(ValueError, match="RRF constant k must be positive"):
        reciprocal_rank_fusion([], k=-10)


def test_rrf_top_k_limiting():
    """Verifies top_k parameter truncates the fused candidate list."""
    list1 = [{"chunk_id": f"chunk_{i}", "rank": i, "source": "s1"} for i in range(1, 10)]
    fused = reciprocal_rank_fusion([list1], k=60, top_k=3)
    assert len(fused) == 3
    assert [f["rank"] for f in fused] == [1, 2, 3]


def test_rrf_empty_lists():
    """Verifies graceful handling of empty lists."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_preserves_metadata():
    """Verifies metadata and document IDs are retained through fusion."""
    item = {
        "chunk_id": 42,
        "document_id": 99,
        "content": "Enterprise security policy",
        "rank": 1,
        "source": "vector",
        "metadata": {"document_name": "sec.pdf", "page_number": 3, "section": "Access"},
    }
    fused = reciprocal_rank_fusion([[item]], k=60)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == 42
    assert fused[0]["document_id"] == 99
    assert fused[0]["content"] == "Enterprise security policy"
    assert fused[0]["metadata"]["document_name"] == "sec.pdf"
    assert fused[0]["metadata"]["page_number"] == 3
    assert fused[0]["metadata"]["section"] == "Access"
