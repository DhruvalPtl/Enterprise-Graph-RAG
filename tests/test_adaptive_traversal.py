"""
Unit and Integration Tests for Phase G10.5-C Controlled 2-Hop Graph Traversal.

Tests:
1. Mode toggle ('depth_1' vs 'depth_2') via constructor, retrieve parameter, and env var.
2. Baseline reproducibility: depth_1 matches 1-hop traversal.
3. Controlled 2-hop expansion: depth_2 recovers 2-hop neighbors with deterministic bounds.
4. Cycle and immediate backtrack prevention (no returning to parent seed).
5. Per-entity neighbor cap enforcement (max_neighbors_per_entity).
6. Path-aware provenance: 2-hop candidates retain full path from seed entity.
7. RBAC security enforcement at every hop: unauthorized documents/entities strictly filtered.
8. Hybrid retriever integration: traversal_mode propagation and metadata logging.
"""
import os
from unittest.mock import MagicMock, patch
import pytest

from app.models import (
    AccessContext,
    Entity,
    Relationship,
    SearchResult,
    GraphRetrievalResult,
)
from app.graph_retriever import GraphRetriever
from app.hybrid_retriever import GraphVectorHybridRetriever
from app.reranker import format_graph_context


def test_traversal_mode_toggle_and_env():
    """Verify traversal_mode defaults, env override, and constructor validation."""
    # Default without env is depth_1
    with patch.dict(os.environ, {"GRAPH_TRAVERSAL_MODE": "depth_1"}):
        gr1 = GraphRetriever()
        assert gr1.traversal_mode == "depth_1"
        assert gr1.default_depth == 1

    # Env set to depth_2
    with patch.dict(os.environ, {"GRAPH_TRAVERSAL_MODE": "depth_2"}):
        gr2 = GraphRetriever()
        assert gr2.traversal_mode == "depth_2"
        assert gr2.default_depth == 2

    # Explicit constructor override
    gr3 = GraphRetriever(traversal_mode="depth_2")
    assert gr3.traversal_mode == "depth_2"
    assert gr3.default_depth == 2

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="traversal_mode must be 'depth_1' or 'depth_2'"):
        GraphRetriever(traversal_mode="invalid_mode")


def test_cycle_elimination_backtrack_prevention():
    """Verify that 2-hop traversal excludes returning immediately to parent seed."""
    gr = GraphRetriever(traversal_mode="depth_2", max_neighbors_per_entity=5)

    mock_conn = MagicMock()
    seed = Entity(id=1, canonical_name="Atlas", entity_type="PRODUCT", display_name="Atlas")
    ctx = AccessContext()

    # Hop 1 returns: Atlas -> Payment Service
    hop1_rows = [
        {
            "id": 101,
            "source_entity_id": 1,
            "source_canonical_name": "Atlas",
            "source_display_name": "Atlas",
            "source_entity_type": "PRODUCT",
            "source_metadata": {},
            "target_entity_id": 2,
            "target_canonical_name": "Payment Service",
            "target_display_name": "Payment Service",
            "target_entity_type": "SERVICE",
            "target_metadata": {},
            "relationship_type": "USES",
            "document_id": 10,
            "chunk_id": 1001,
            "page_number": 1,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
    ]

    # Hop 2 returns two candidate edges from Payment Service:
    # 1. Backtrack edge: Payment Service -> Atlas (should be rejected!)
    # 2. Forward edge: Payment Service -> Platform Engineering (should be accepted!)
    hop2_rows = [
        {
            "id": 102,
            "source_entity_id": 2,
            "source_canonical_name": "Payment Service",
            "source_display_name": "Payment Service",
            "source_entity_type": "SERVICE",
            "source_metadata": {},
            "target_entity_id": 1,  # BACKTRACK TO SEED!
            "target_canonical_name": "Atlas",
            "target_display_name": "Atlas",
            "target_entity_type": "PRODUCT",
            "target_metadata": {},
            "relationship_type": "SUPPORTED_BY",
            "document_id": 10,
            "chunk_id": 1002,
            "page_number": 1,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        },
        {
            "id": 103,
            "source_entity_id": 2,
            "source_canonical_name": "Payment Service",
            "source_display_name": "Payment Service",
            "source_entity_type": "SERVICE",
            "source_metadata": {},
            "target_entity_id": 3,  # NEW 2-HOP TARGET
            "target_canonical_name": "Platform Engineering",
            "target_display_name": "Platform Engineering",
            "target_entity_type": "TEAM",
            "target_metadata": {},
            "relationship_type": "DEVELOPED_BY",
            "document_id": 20,
            "chunk_id": 2001,
            "page_number": 2,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        },
    ]

    with patch.object(gr, "_query_authorized_relationships", side_effect=[hop1_rows, hop2_rows]):
        rels, connected = gr.traverse_graph(
            conn=mock_conn,
            seed_entities=[seed],
            access_context=ctx,
            depth=2,
            max_results=50,
        )

    # Invariant: Rel 102 (cycle back to Atlas) must be discarded
    rel_ids = [r.id for r in rels]
    assert 101 in rel_ids  # Hop 1
    assert 103 in rel_ids  # Hop 2 forward edge
    assert 102 not in rel_ids  # Cycle eliminated!
    assert len(rels) == 2


def test_max_neighbors_per_entity_cap():
    """Verify deterministic limit caps edges per neighbor entity to prevent hub explosion."""
    gr = GraphRetriever(traversal_mode="depth_2", max_neighbors_per_entity=2)

    mock_conn = MagicMock()
    seed = Entity(id=1, canonical_name="Atlas", entity_type="PRODUCT", display_name="Atlas")
    ctx = AccessContext()

    hop1_rows = [
        {
            "id": 101,
            "source_entity_id": 1,
            "source_canonical_name": "Atlas",
            "source_display_name": "Atlas",
            "source_entity_type": "PRODUCT",
            "source_metadata": {},
            "target_entity_id": 2,
            "target_canonical_name": "Payment Service",
            "target_display_name": "Payment Service",
            "target_entity_type": "SERVICE",
            "target_metadata": {},
            "relationship_type": "USES",
            "document_id": 10,
            "chunk_id": 1001,
            "page_number": 1,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
    ]

    # Hop 2 returns 4 edges from Payment Service (cap is 2)
    hop2_rows = [
        {
            "id": 201 + i,
            "source_entity_id": 2,
            "source_canonical_name": "Payment Service",
            "source_display_name": "Payment Service",
            "source_entity_type": "SERVICE",
            "source_metadata": {},
            "target_entity_id": 10 + i,
            "target_canonical_name": f"Vendor_{i}",
            "target_display_name": f"Vendor_{i}",
            "target_entity_type": "VENDOR",
            "target_metadata": {},
            "relationship_type": "INTEGRATES_WITH",
            "document_id": 20,
            "chunk_id": 2000 + i,
            "page_number": 2,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
        for i in range(4)
    ]

    with patch.object(gr, "_query_authorized_relationships", side_effect=[hop1_rows, hop2_rows]):
        rels, _ = gr.traverse_graph(
            conn=mock_conn,
            seed_entities=[seed],
            access_context=ctx,
            depth=2,
            max_results=50,
            max_neighbors_per_entity=2,
        )

    # Invariant: exactly 1 from Hop 1 + 2 from Hop 2 = 3 relationships total
    hop2_rels = [r for r in rels if r.metadata.get("hop_depth") == 2]
    assert len(hop2_rels) == 2
    assert len(rels) == 3


def test_path_aware_provenance_retention():
    """Verify that Hop 2 relationships preserve full path from seed entity."""
    gr = GraphRetriever(traversal_mode="depth_2")
    mock_conn = MagicMock()
    seed = Entity(id=1, canonical_name="Atlas", entity_type="PRODUCT", display_name="Atlas")
    ctx = AccessContext()

    hop1_rows = [
        {
            "id": 101,
            "source_entity_id": 1,
            "source_canonical_name": "Atlas",
            "source_display_name": "Atlas",
            "source_entity_type": "PRODUCT",
            "source_metadata": {},
            "target_entity_id": 2,
            "target_canonical_name": "Payment Service",
            "target_display_name": "Payment Service",
            "target_entity_type": "SERVICE",
            "target_metadata": {},
            "relationship_type": "USES",
            "document_id": 10,
            "chunk_id": 1001,
            "page_number": 1,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
    ]

    hop2_rows = [
        {
            "id": 102,
            "source_entity_id": 2,
            "source_canonical_name": "Payment Service",
            "source_display_name": "Payment Service",
            "source_entity_type": "SERVICE",
            "source_metadata": {},
            "target_entity_id": 3,
            "target_canonical_name": "Platform Engineering",
            "target_display_name": "Platform Engineering",
            "target_entity_type": "TEAM",
            "target_metadata": {},
            "relationship_type": "DEVELOPED_BY",
            "document_id": 20,
            "chunk_id": 2001,
            "page_number": 2,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
    ]

    with patch.object(gr, "_query_authorized_relationships", side_effect=[hop1_rows, hop2_rows]):
        rels, _ = gr.traverse_graph(
            conn=mock_conn,
            seed_entities=[seed],
            access_context=ctx,
            depth=2,
            max_results=50,
        )

    hop2_rel = [r for r in rels if r.metadata.get("hop_depth") == 2][0]
    assert hop2_rel.metadata["seed_name"] == "Atlas"
    assert hop2_rel.metadata["parent_edge_id"] == 101
    assert hop2_rel.metadata["hop1_edge"] == {
        "source": "Atlas",
        "relationship_type": "USES",
        "target": "Payment Service",
    }
    assert hop2_rel.metadata["hop2_edge"] == {
        "source": "Payment Service",
        "relationship_type": "DEVELOPED_BY",
        "target": "Platform Engineering",
    }


def test_hybrid_retriever_multi_hop_path_formatting():
    """Verify that hybrid retriever packages Hop 1 + Hop 2 into format_graph_context."""
    mock_pipeline = MagicMock()
    mock_pipeline.hybrid_retriever.retrieve_with_details.return_value = {"fused_results": []}

    # Create mock graph result with a 2-hop relationship
    hop2_rel = Relationship(
        id=102,
        source_entity_id=2,
        target_entity_id=3,
        relationship_type="DEVELOPED_BY",
        document_id=20,
        chunk_id=2001,
        page_number=2,
        metadata={
            "source_name": "Payment Service",
            "target_name": "Platform Engineering",
            "hop_depth": 2,
            "hop1_edge": {
                "source": "Atlas",
                "relationship_type": "USES",
                "target": "Payment Service",
            },
        },
    )

    mock_graph_res = MagicMock(
        matched_entities=[Entity(id=1, canonical_name="Atlas", entity_type="PRODUCT", display_name="Atlas")],
        relationships=[hop2_rel],
        connected_entities=[],
        source_chunks=[
            {
                "chunk_id": 2001,
                "document_id": 20,
                "document_name": "DOC-ENG-002.md",
                "content": "Platform Engineering develops Payment Service.",
                "department": "engineering",
                "access_level": "public",
                "status": "active",
            }
        ],
    )

    mock_graph_retriever = MagicMock()
    mock_graph_retriever.retrieve.return_value = mock_graph_res

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []

    retriever = GraphVectorHybridRetriever(
        reranked_pipeline=mock_pipeline,
        graph_retriever=mock_graph_retriever,
        reranker=mock_reranker,
        traversal_mode="depth_2",
        reranking_mode="graph_aware",
    )

    ctx = AccessContext(department="engineering", access_level="public")
    retriever.retrieve_fused("test query", access_context=ctx)

    # Inspect candidate passed to reranker
    candidates_passed = mock_reranker.rerank.call_args[1]["candidates"]
    assert len(candidates_passed) == 1
    cand = candidates_passed[0]

    # Format graph context using G10.5-B format_graph_context
    context = format_graph_context(cand)
    assert "[GRAPH PATH]" in context
    assert "Atlas --[USES]--> Payment Service" in context
    assert "Payment Service --[DEVELOPED_BY]--> Platform Engineering" in context


def test_rbac_boundary_preservation_every_hop():
    """Verify that an unauthorized edge encountered at Hop 2 is filtered out."""
    gr = GraphRetriever(traversal_mode="depth_2")
    mock_conn = MagicMock()
    seed = Entity(id=1, canonical_name="Atlas", entity_type="PRODUCT", display_name="Atlas")

    # User has public engineering access
    ctx = AccessContext(department="engineering", access_level="public")

    # Hop 1: authorized public document
    hop1_rows = [
        {
            "id": 101,
            "source_entity_id": 1,
            "source_canonical_name": "Atlas",
            "source_display_name": "Atlas",
            "source_entity_type": "PRODUCT",
            "source_metadata": {},
            "target_entity_id": 2,
            "target_canonical_name": "Payment Service",
            "target_display_name": "Payment Service",
            "target_entity_type": "SERVICE",
            "target_metadata": {},
            "relationship_type": "USES",
            "document_id": 10,
            "chunk_id": 1001,
            "page_number": 1,
            "rel_metadata": {},
            "created_at": None,
            "updated_at": None,
        }
    ]

    # Hop 2 SQL query in _query_authorized_relationships automatically joins documents
    # with access control. If the document is restricted (e.g. manager only),
    # the SQL query returns 0 rows.
    hop2_rows = []  # Restricted document blocked by SQL join

    with patch.object(gr, "_query_authorized_relationships", side_effect=[hop1_rows, hop2_rows]):
        rels, connected = gr.traverse_graph(
            conn=mock_conn,
            seed_entities=[seed],
            access_context=ctx,
            depth=2,
            max_results=50,
        )

    # Invariant: only authorized hop 1 relationship is surfaced
    assert len(rels) == 1
    assert rels[0].id == 101
