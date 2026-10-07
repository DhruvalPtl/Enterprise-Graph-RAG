"""
Unit and Integration Tests for Phase G10.5-A Entity Linking.

Tests:
1. Normalization & alphanumeric code regex extraction.
2. Canonical alias and role mapping.
3. Semantic candidate generation and confidence threshold gating.
4. Noise suppression for generic search stopwords.
5. RBAC clearance verification and zero-leakage pruning.
6. GraphRetriever runtime mode toggle ('baseline' vs 'improved').
7. Observability telemetry fields on GraphRetrievalResult.
"""
import pytest
import psycopg

from app.db import get_connection
from app.entity_linker import EntityLinker
from app.graph_retriever import GraphRetriever
from app.models import AccessContext, Entity, GraphRetrievalResult


@pytest.fixture(scope="module")
def db_conn():
    conn = get_connection()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def entity_linker():
    return EntityLinker(semantic_threshold=0.65, enable_semantic=True)


@pytest.fixture(scope="module")
def improved_retriever(entity_linker):
    return GraphRetriever(entity_linking_mode="improved", entity_linker=entity_linker)


@pytest.fixture(scope="module")
def baseline_retriever():
    return GraphRetriever(entity_linking_mode="baseline")


# ==============================================================================
# 1. Alphanumeric Code & Exact Identifier Tests
# ==============================================================================
def test_exact_code_extraction(entity_linker):
    canons, audit = entity_linker.extract_candidates("Review service PAY-SVC-017 and customer CUST-1042.")
    assert "payment service" in canons
    assert "acme corp" in canons
    assert canons["payment service"][0] == "code_alias"
    assert canons["acme corp"][0] == "code_alias"


def test_atlas_code_extraction(entity_linker):
    canons, audit = entity_linker.extract_candidates("Status of ATLAS-PROD-001 deployment.")
    assert "product atlas" in canons
    assert canons["product atlas"][0] == "code_alias"


# ==============================================================================
# 2. Canonical Alias & Acronym Tests
# ==============================================================================
def test_organizational_acronyms(entity_linker):
    canons, _ = entity_linker.extract_candidates("Who leads the SOC and who is in HR?")
    assert "security operations center" in canons
    assert "human resources" in canons


def test_role_and_executive_aliases(entity_linker):
    canons, _ = entity_linker.extract_candidates("The CISO met with the Lead SRE and the VP of Engineering.")
    assert "david ross" in canons
    assert "alex rivera" in canons
    assert "elena vance" in canons


def test_policy_aliases(entity_linker):
    canons, _ = entity_linker.extract_candidates("What are the remote work rules, PTO limits, and per diem rates?")
    assert "remote work policy" in canons
    assert "leave policy" in canons
    assert "travel and expense policy" in canons


# ==============================================================================
# 3. Currency & Metric Tests
# ==============================================================================
def test_currency_and_metric_patterns(entity_linker):
    canons, _ = entity_linker.extract_candidates("Verify the $100/day meal cap and the $50,000 procurement limit.")
    assert "$100/day meal cap" in canons
    assert "$50,000 procurement limit" in canons


def test_sla_availability_pattern(entity_linker):
    canons, _ = entity_linker.extract_candidates("Does Atlas maintain 99.95% availability?")
    assert "99.95% availability" in canons
    assert "product atlas" in canons


# ==============================================================================
# 4. Semantic Fallback & Noise Gating Tests
# ==============================================================================
def test_semantic_fallback_natural_phrasing():
    # Test semantic candidate generation with tuned threshold on natural language description
    linker = EntityLinker(semantic_threshold=0.60, enable_semantic=True)
    canons, audit = linker.extract_candidates("What platform serves as the distributed messaging backbone?")
    assert len(canons) > 0
    assert "kafka event bus" in canons
    assert canons["kafka event bus"][0] == "semantic"
    assert canons["kafka event bus"][1] >= 0.60


def test_generic_noise_words_not_promoted(entity_linker):
    # Isolated generic query tokens should not trigger spurious entities
    canons, _ = entity_linker.extract_candidates("code window event policy")
    assert len(canons) == 0


# ==============================================================================
# 5. RBAC Clearance & Security Tests
# ==============================================================================
def test_rbac_security_admin_access(entity_linker, db_conn):
    ctx_admin = AccessContext(department="security", access_level="admin")
    ents, tel = entity_linker.link_entities(
        conn=db_conn,
        query="What happened during incident INC-2026-021?",
        access_context=ctx_admin,
    )
    assert any(e.canonical_name == "incident inc-2026-021" for e in ents)
    assert tel["authorized_count"] >= 1


def test_rbac_security_unprivileged_denied_access(entity_linker, db_conn):
    # An engineering employee without security admin clearance asking about confidential token incident
    ctx_unprivileged = AccessContext(department="engineering", access_level="employee")
    ents, tel = entity_linker.link_entities(
        conn=db_conn,
        query="What happened during incident INC-2026-021?",
        access_context=ctx_unprivileged,
    )
    # Incident INC-2026-021 is documented in DOC-SEC-004 (admin clearance). Should be pruned!
    assert not any(e.canonical_name == "incident inc-2026-021" for e in ents)


# ==============================================================================
# 6. GraphRetriever Integration & Mode Toggle Tests
# ==============================================================================
def test_graph_retriever_mode_toggle(improved_retriever, baseline_retriever, db_conn):
    ctx = AccessContext(department="engineering", access_level="employee")
    query = "What microservices does ATLAS-PROD-001 depend on?"

    # Baseline mode: fails to parse ATLAS-PROD-001 code
    base_res = baseline_retriever.retrieve(query=query, access_context=ctx, conn=db_conn)
    assert base_res.retrieval_metadata["entity_linking_mode"] == "baseline"

    # Improved mode: resolves ATLAS-PROD-001 -> product atlas and retrieves subgraphs
    imp_res = improved_retriever.retrieve(query=query, access_context=ctx, conn=db_conn)
    assert imp_res.retrieval_metadata["entity_linking_mode"] == "improved"
    assert imp_res.retrieval_metadata["entity_linking_method"] == "code_alias"
    assert any(e.canonical_name == "product atlas" for e in imp_res.matched_entities)
    assert len(imp_res.relationships) > 0


def test_telemetry_fields_presence(improved_retriever, db_conn):
    ctx = AccessContext(department="public", access_level="public")
    res = improved_retriever.retrieve(query="NovaTech Systems headquarters in Seattle", access_context=ctx, conn=db_conn)
    meta = res.retrieval_metadata
    assert "entity_linking_mode" in meta
    assert "entity_linking_method" in meta
    assert "candidate_entities" in meta
    assert "linking_execution_time_ms" in meta
    assert meta["linking_execution_time_ms"] >= 0.0
