"""
Entity Linking Engine (Phase G10.5-A).

Provides precision-first, modular entity linking over PostgreSQL knowledge graph:
1. Normalizes user queries across casing, punctuation, and hyphen/underscore variations.
2. Extracts exact alphanumeric enterprise codes (e.g. ATLAS-PROD-001, PAY-SVC-017, CUST-1042).
3. Resolves canonical aliases, acronyms, and organizational roles (e.g. SOC, HR, SRE, CISO).
4. Matches canonical names and display names against PostgreSQL entities.
5. Employs semantic candidate generation via sentence-transformers/all-MiniLM-L6-v2 with strict
   confidence gating (threshold >= 0.65) when exact and alias matches yield no candidates.
6. Enforces pre-retrieval RBAC access control by pruning entities that exist exclusively in
   confidential documents above caller clearance or outside caller department.
7. Emits structured observability telemetry recording linking methods, candidate scores, and timings.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging
import os
import re
import time

import numpy as np
import psycopg
from psycopg.rows import dict_row

from app.models import AccessContext, Entity

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENTITIES_PATH = BASE_DIR / "data" / "evaluation" / "enterprise_corpus" / "entities.json"

# Common noise words to prevent false-positive semantic promotions
GENERIC_STOP_PHRASES: Set[str] = {
    "code", "window", "event", "policy", "standard", "report", "service",
    "system", "data", "information", "benchmark", "procedure", "document",
}


class EntityLinker:
    """
    Precision-first modular entity linker for Graph RAG (Phase G10.5-A).
    """

    def __init__(
        self,
        entities_path: Optional[Path] = None,
        semantic_threshold: float = 0.65,
        enable_semantic: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            entities_path: Path to entities.json.
            semantic_threshold: Cosine similarity cutoff for semantic fallback (default 0.65).
            enable_semantic: Whether to enable MiniLM semantic candidate generation.
            embedding_model_name: SentenceTransformers model identifier.
        """
        self.entities_path = entities_path or DEFAULT_ENTITIES_PATH
        self.semantic_threshold = semantic_threshold
        self.enable_semantic = enable_semantic
        self.embedding_model_name = embedding_model_name

        self._embed_model = None
        self._corpus_entities: List[Dict[str, Any]] = []
        self._canonical_keys: List[Dict[str, Any]] = []
        self._entity_embeddings: Optional[np.ndarray] = None
        self._alias_map: Dict[str, str] = {}
        self._code_pattern = re.compile(r'\b([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)\b')

        self._init_alias_map()
        if self.enable_semantic:
            self._init_embeddings()

    def _init_alias_map(self) -> None:
        """Initializes canonical alias dictionary directly from enterprise corpus metadata."""
        # 1. Base enterprise acronyms and common roles
        self._alias_map = {
            # Organizations & Departments
            "novatech": "novatech systems",
            "novatech systems": "novatech systems",
            "platform engineering": "platform engineering",
            "cloudops": "cloudops team",
            "cloudops team": "cloudops team",
            "sre": "cloudops team",
            "site reliability engineering": "cloudops team",
            "security operations center": "security operations center",
            "soc": "security operations center",
            "finance": "finance department",
            "finance department": "finance department",
            "human resources": "human resources",
            "hr": "human resources",
            "people operations": "human resources",

            # Key Personnel & Roles
            "elena vance": "elena vance",
            "vp of engineering": "elena vance",
            "vp engineering": "elena vance",
            "marcus chen": "marcus chen",
            "principal architect": "marcus chen",
            "tech lead": "marcus chen",
            "priya sharma": "priya sharma",
            "director of human resources": "priya sharma",
            "director of hr": "priya sharma",
            "hr director": "priya sharma",
            "david ross": "david ross",
            "chief information security officer": "david ross",
            "ciso": "david ross",
            "sarah jenkins": "sarah jenkins",
            "corporate finance director": "sarah jenkins",
            "finance director": "sarah jenkins",
            "alex rivera": "alex rivera",
            "lead site reliability engineer": "alex rivera",
            "lead sre": "alex rivera",
            "incident commander": "alex rivera",

            # Products & Core Services
            "product atlas": "product atlas",
            "atlas": "product atlas",
            "product orion": "product orion",
            "orion": "product orion",
            "novaportal": "novaportal",
            "employee portal": "novaportal",
            "intranet portal": "novaportal",
            "payment service": "payment service",
            "payment gateway": "payment service",
            "payment processing": "payment service",
            "auth service": "auth service",
            "authentication service": "auth service",
            "notification gateway": "notification gateway",
            "notification service": "notification gateway",
            "kafka": "kafka event bus",
            "kafka event bus": "kafka event bus",
            "event bus": "kafka event bus",
            "postgresql": "postgresql cluster",
            "postgres": "postgresql cluster",
            "postgresql cluster": "postgresql cluster",

            # External Customers & Vendors
            "acme corp": "acme corp",
            "acme": "acme corp",
            "globalfin holdings": "globalfin holdings",
            "globalfin": "globalfin holdings",
            "datacloud inc": "datacloud inc",
            "datacloud": "datacloud inc",
            "fastcdn networks": "fastcdn networks",
            "fastcdn": "fastcdn networks",
            "apex payments": "apex payments",
            "apex": "apex payments",

            # Corporate Policies & Standards
            "remote work policy": "remote work policy",
            "remote work": "remote work policy",
            "telecommuting": "remote work policy",
            "work from home": "remote work policy",
            "home workspace": "remote work policy",
            "leave policy": "leave policy",
            "leave": "leave policy",
            "paid time off": "leave policy",
            "pto": "leave policy",
            "vacation": "leave policy",
            "sick leave": "leave policy",
            "parental leave": "leave policy",
            "corporate procurement policy": "corporate procurement policy",
            "procurement policy": "corporate procurement policy",
            "procurement": "corporate procurement policy",
            "purchasing": "corporate procurement policy",
            "vendor spending": "corporate procurement policy",
            "travel and expense policy": "travel and expense policy",
            "travel & expense": "travel and expense policy",
            "travel policy": "travel and expense policy",
            "expense policy": "travel and expense policy",
            "travel and expense": "travel and expense policy",
            "per-diem": "travel and expense policy",
            "per diem": "travel and expense policy",
            "food expenditures": "travel and expense policy",
            "enterprise access control policy": "enterprise access control policy",
            "access control policy": "enterprise access control policy",
            "access control": "enterprise access control policy",
            "zero trust": "enterprise access control policy",
            "data classification standard": "data classification standard",
            "data classification": "data classification standard",
            "classification standard": "data classification standard",

            # Security & Operations Incidents
            "payment gateway outage": "incident inc-2026-014",
            "token leak": "incident inc-2026-021",
            "authentication token leak": "incident inc-2026-021",

            # Geographic Hubs & Datacenters
            "seattle": "seattle",
            "mumbai": "mumbai",
            "london": "london",
            "frankfurt": "frankfurt",
            "virginia": "virginia",

            # Critical Metrics & Ceilings
            "99.95%": "99.95% availability",
            "99.95% availability": "99.95% availability",
            "p99 50ms": "p99 50ms latency",
            "p99 latency": "p99 50ms latency",
            "50ms latency": "p99 50ms latency",
            "$100/day": "$100/day meal cap",
            "$100 daily": "$100/day meal cap",
            "meal cap": "$100/day meal cap",
            "meal allowance": "$100/day meal cap",
            "$50,000": "$50,000 procurement limit",
            "$50,000 procurement": "$50,000 procurement limit",
        }

        # 2. Ingest corpus entities file if available
        if self.entities_path.exists():
            try:
                with open(self.entities_path, "r", encoding="utf-8") as f:
                    self._corpus_entities = json.load(f)

                for e in self._corpus_entities:
                    cname = e["canonical_name"].lower()
                    name = e["name"].lower()
                    self._alias_map[cname] = cname
                    self._alias_map[name] = cname

                    # Scan descriptions and names for alphanumeric codes
                    for m in self._code_pattern.finditer(e.get("description", "")):
                        self._alias_map[m.group(1).lower()] = cname
                    for m in self._code_pattern.finditer(e.get("name", "")):
                        self._alias_map[m.group(1).lower()] = cname

            except Exception as ex:
                logger.warning(f"Could not load entities from {self.entities_path}: {ex}")

    def _init_embeddings(self) -> None:
        """Pre-computes compact entity embeddings for semantic candidate generation."""
        if not self._corpus_entities:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embedding_model_name)
            self._canonical_keys = list(self._corpus_entities)
            texts = [
                f"{e['name']}: {e.get('description', '')} ({e.get('type', '')})"
                for e in self._canonical_keys
            ]
            self._entity_embeddings = self._embed_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as ex:
            logger.warning(f"Failed to initialize SentenceTransformer for entity linking: {ex}")
            self._embed_model = None
            self._entity_embeddings = None

    def extract_candidates(
        self,
        query: str,
    ) -> Tuple[Dict[str, Tuple[str, float]], List[Dict[str, Any]]]:
        """
        Extracts entity candidates from query string.

        Returns:
            Tuple of (matched_canons_map, candidate_audit_list).
            matched_canons_map: {canonical_name: (method, confidence_score)}
        """
        clean_q = query.lower()
        matched_canons: Dict[str, Tuple[str, float]] = {}
        candidate_audit: List[Dict[str, Any]] = []

        # 1. Alphanumeric Identifier Matching (regex)
        found_codes = self._code_pattern.findall(query.upper())
        for code in found_codes:
            code_lower = code.lower()
            if code_lower in self._alias_map:
                canon = self._alias_map[code_lower]
                if canon not in matched_canons:
                    matched_canons[canon] = ("code_alias", 1.0)
                    candidate_audit.append({
                        "name": canon,
                        "method": "code_alias",
                        "trigger": code,
                        "score": 1.0,
                    })

        # 2. Alias and Exact Name Matching (longest first for specificity)
        sorted_terms = sorted(self._alias_map.keys(), key=len, reverse=True)
        for term in sorted_terms:
            canon = self._alias_map[term]
            if canon in matched_canons:
                continue

            # Special characters / punctuation / symbols boundary handling
            if any(c in term for c in "$%0123456789-"):
                if term in clean_q:
                    matched_canons[canon] = ("exact", 1.0)
                    candidate_audit.append({
                        "name": canon,
                        "method": "exact",
                        "trigger": term,
                        "score": 1.0,
                    })
            else:
                if re.search(r'\b' + re.escape(term) + r'\b', clean_q):
                    method = "exact" if term == canon else "alias"
                    score = 1.0 if method == "exact" else 0.98
                    matched_canons[canon] = (method, score)
                    candidate_audit.append({
                        "name": canon,
                        "method": method,
                        "trigger": term,
                        "score": score,
                    })

        # 3. Semantic Fallback (MiniLM) if zero exact candidates matched
        if len(matched_canons) == 0 and self.enable_semantic and self._embed_model is not None and self._entity_embeddings is not None:
            # Check query length
            clean_tokens = [w for w in re.sub(r'[^\w\s]', ' ', clean_q).split() if len(w) >= 3]
            if len(clean_tokens) >= 2 and not all(t in GENERIC_STOP_PHRASES for t in clean_tokens):
                q_emb = self._embed_model.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                sims = np.dot(self._entity_embeddings, q_emb)
                top_idx = int(np.argmax(sims))
                score = float(sims[top_idx])
                if score >= self.semantic_threshold:
                    canon = self._canonical_keys[top_idx]["canonical_name"].lower()
                    matched_canons[canon] = ("semantic", round(score, 4))
                    candidate_audit.append({
                        "name": canon,
                        "method": "semantic",
                        "trigger": query,
                        "score": round(score, 4),
                    })

        return matched_canons, candidate_audit

    def link_entities(
        self,
        conn: psycopg.Connection,
        query: str,
        access_context: Optional[AccessContext] = None,
        limit: int = 10,
    ) -> Tuple[List[Entity], Dict[str, Any]]:
        """
        Links query to PostgreSQL Entity records with RBAC clearance filtering.

        Args:
            conn: PostgreSQL connection.
            query: User search query.
            access_context: Caller authorization context.
            limit: Maximum seed entities to return.

        Returns:
            Tuple of (authorized_entities, telemetry_metadata).
        """
        t0 = time.perf_counter()
        if not query or not query.strip():
            return [], {
                "entity_linking_method": "none",
                "candidate_count": 0,
                "authorized_count": 0,
                "execution_time_ms": 0.0,
                "candidates": [],
            }

        ctx = access_context or AccessContext()

        # 1. Extract entity candidate canonical names
        matched_canons, candidate_audit = self.extract_candidates(query)
        if not matched_canons:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return [], {
                "entity_linking_method": "none",
                "candidate_count": 0,
                "authorized_count": 0,
                "execution_time_ms": elapsed_ms,
                "candidates": [],
            }

        # 2. Query PostgreSQL to fetch entity records for matched canonical names
        candidate_canons = list(matched_canons.keys())
        sql_fetch = """
        SELECT id, canonical_name, entity_type, display_name, metadata, created_at, updated_at
        FROM entities
        WHERE LOWER(canonical_name) = ANY(%s) OR LOWER(display_name) = ANY(%s);
        """
        candidate_entities: List[Entity] = []
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_fetch, (candidate_canons, candidate_canons))
            for row in cur.fetchall():
                candidate_entities.append(Entity.from_dict(row))

        # 3. Enforce RBAC Pre-Retrieval Authorization Filter
        # If caller is not admin, verify that the entity has at least one incident relationship
        # in a document authorized under the caller's access level and department.
        authorized_entities: List[Entity] = []
        if ctx.is_admin:
            authorized_entities = candidate_entities
        elif candidate_entities:
            allowed_levels = ctx.allowed_access_levels()
            eids = [e.id for e in candidate_entities if e.id is not None]
            sql_rbac = """
            SELECT DISTINCT r.source_entity_id, r.target_entity_id
            FROM relationships r
            JOIN documents d ON r.document_id = d.id
            WHERE (r.source_entity_id = ANY(%s) OR r.target_entity_id = ANY(%s))
              AND (%s OR d.status = 'active')
              AND (%s OR d.access_level = ANY(%s))
              AND (%s OR d.department = 'public' OR d.department = %s);
            """
            params = (
                eids,
                eids,
                ctx.include_archived,
                ctx.is_admin,
                allowed_levels,
                ctx.is_admin,
                ctx.department,
            )
            with conn.cursor() as cur:
                cur.execute(sql_rbac, params)
                auth_eids = set()
                for row in cur.fetchall():
                    auth_eids.add(row[0])
                    auth_eids.add(row[1])
            authorized_entities = [e for e in candidate_entities if e.id in auth_eids]

        # Truncate and sort by confidence / specificity
        final_entities = authorized_entities[:limit]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Primary method tag
        methods = {audit["method"] for audit in candidate_audit}
        if "semantic" in methods and len(methods) == 1:
            primary_method = "semantic"
        elif "code_alias" in methods:
            primary_method = "code_alias"
        elif "alias" in methods:
            primary_method = "alias"
        elif "exact" in methods:
            primary_method = "exact"
        elif len(methods) > 1:
            primary_method = "hybrid"
        else:
            primary_method = "none"

        telemetry = {
            "entity_linking_method": primary_method,
            "candidate_count": len(candidate_entities),
            "authorized_count": len(authorized_entities),
            "execution_time_ms": elapsed_ms,
            "candidates": candidate_audit,
        }
        return final_entities, telemetry
