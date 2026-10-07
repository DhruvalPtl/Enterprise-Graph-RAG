"""
Graph Retrieval Engine (Phase G3).

Provides production-oriented, bounded graph retrieval over PostgreSQL knowledge graph:
1. Identifies useful entity candidate mentions and keywords from natural language queries.
2. Searches PostgreSQL entities (canonical_name, display_name) to locate seed entities.
3. Bounded graph traversal (depth 1 or 2) expanding incident semantic relationships.
4. Preserves complete source provenance (chunk_id, document_id, page_number).
5. Strictly enforces pre-retrieval document access control (AccessContext).
6. Deduplicates entities, relationships, and source chunks while capping results
   to prevent context explosion.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import re
import time
import psycopg
from psycopg.rows import dict_row

from app.db import get_connection
from app.models import (
    AccessContext,
    Entity,
    Relationship,
    GraphRetrievalResult,
)


# Common English stop words excluded from entity candidate n-grams
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Question query verbs and meta words
    "tell", "show", "find", "give", "describe", "explain", "summarize", "list",
    "report", "results", "findings", "information", "data", "benchmark", "benchmarks",
}


class GraphRetriever:
    """
    Retrieves relevant subgraphs and provenance chunks from PostgreSQL knowledge graph.
    """

    def __init__(
        self,
        default_depth: int = 1,
        default_max_results: int = 50,
        default_max_seeds: int = 10,
        entity_linking_mode: Optional[str] = None,
        entity_linker: Optional[Any] = None,
        traversal_mode: Optional[str] = None,
        max_neighbors_per_entity: int = 5,
    ):
        """
        Args:
            default_depth: Default traversal depth (1 or 2).
            default_max_results: Default maximum relationships to return.
            default_max_seeds: Default maximum seed entities to match from query.
            entity_linking_mode: Mode for entity linking ('baseline' or 'improved').
            entity_linker: Optional pre-initialized EntityLinker instance.
            traversal_mode: Mode for graph traversal ('depth_1' or 'depth_2').
            max_neighbors_per_entity: Max Hop 2 relationships allowed per 1-hop neighbor.
        """
        t_mode = (traversal_mode or os.getenv("GRAPH_TRAVERSAL_MODE", "depth_1")).strip().lower()
        if t_mode not in ("depth_1", "depth_2"):
            raise ValueError(f"traversal_mode must be 'depth_1' or 'depth_2', got '{t_mode}'.")
        self.traversal_mode = t_mode

        # If traversal_mode is explicitly depth_2 and default_depth was not overridden, use depth 2
        if t_mode == "depth_2" and default_depth == 1:
            self.default_depth = 2
        else:
            self.default_depth = default_depth

        if self.default_depth not in (1, 2):
            raise ValueError(f"Traversal depth must be 1 or 2, got {self.default_depth}.")
        self.default_max_results = default_max_results
        self.default_max_seeds = default_max_seeds
        self.max_neighbors_per_entity = max_neighbors_per_entity

        mode = entity_linking_mode or os.getenv("ENTITY_LINKING_MODE", "baseline")
        if mode not in ("baseline", "improved"):
            raise ValueError(f"entity_linking_mode must be 'baseline' or 'improved', got '{mode}'.")
        self.entity_linking_mode = mode
        self._entity_linker = entity_linker

    def get_entity_linker(self) -> Any:
        """Lazily initializes and returns the improved EntityLinker instance."""
        if self._entity_linker is None:
            from app.entity_linker import EntityLinker
            self._entity_linker = EntityLinker()
        return self._entity_linker

    def extract_keywords(self, query: str) -> List[str]:
        """
        Identifies useful entity candidate phrases and keywords from a user query.
        Extracts:
        1. Quoted terms (e.g. "Llama 2 70B").
        2. Capitalized or alphanumeric proper noun sequences (e.g. "Stanford University", "Gemini Ultra", "AI Act").
        3. Sliding n-grams (1 to 4 words) omitting stop words.
        4. Cleaned sub-phrases.

        Returns candidate strings sorted by length in descending order.
        """
        if not query or not query.strip():
            return []

        candidates: Set[str] = set()

        # 1. Quoted terms
        quotes = re.findall(r'["\']([^"\']+)["\']', query)
        for q in quotes:
            cleaned = q.strip()
            if len(cleaned) >= 2 and cleaned.lower() not in STOP_WORDS:
                candidates.add(cleaned)

        # 2. Capitalized / titled / alphanumeric sequences (proper nouns, model names, acronyms)
        # Matches e.g. "Stanford University", "Llama 2 70B", "GPT-4", "AI Act", "UniAudio"
        cap_phrases = re.findall(r'[A-Z0-9][a-zA-Z0-9_\-\.]*(?:\s+[A-Z0-9][a-zA-Z0-9_\-\.]*)*', query)
        for p in cap_phrases:
            cleaned = p.strip(" .-,")
            if len(cleaned) >= 2 and cleaned.lower() not in STOP_WORDS:
                candidates.add(cleaned)

        # 3. Clean token-based sliding n-grams (1 to 4 words)
        clean_text = re.sub(r'[^\w\s\-\.]', ' ', query)
        words = clean_text.split()
        n = len(words)
        max_ngram_len = min(4, n)

        for length in range(1, max_ngram_len + 1):
            for i in range(n - length + 1):
                ngram_tokens = words[i:i + length]
                # Skip if all tokens are stop words
                if all(t.lower() in STOP_WORDS for t in ngram_tokens):
                    continue
                ngram = " ".join(ngram_tokens).strip(" .-,")
                if len(ngram) >= 2 and ngram.lower() not in STOP_WORDS:
                    candidates.add(ngram)

        # Sort candidates: prioritize longer, more specific phrases first
        return sorted(candidates, key=lambda s: (len(s), len(s.split())), reverse=True)

    def find_seed_entities(
        self,
        conn: psycopg.Connection,
        query: str,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Searches PostgreSQL entities using canonical_name and display_name
        against extracted query candidates.

        Order of priority:
        1. Exact match on canonical_name or display_name (case-insensitive).
        2. Substring match (ILIKE) on top candidates.
        Matches are deduplicated by entity ID and limited to `limit`.
        """
        candidates = self.extract_keywords(query)
        if not candidates:
            return []

        matched_entities: List[Entity] = []
        seen_ids: Set[int] = set()

        # Phase 1: Exact match query
        lower_candidates = [c.lower() for c in candidates[:30]]
        sql_exact = """
        SELECT id, canonical_name, entity_type, display_name, metadata, created_at, updated_at
        FROM entities
        WHERE LOWER(canonical_name) = ANY(%s) OR LOWER(display_name) = ANY(%s)
        ORDER BY LENGTH(canonical_name) DESC
        LIMIT %s;
        """
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql_exact, (lower_candidates, lower_candidates, limit))
            for row in cur.fetchall():
                eid = row["id"]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    matched_entities.append(Entity.from_dict(row))

        # Phase 2: If exact matches are fewer than limit, execute substring ILIKE fallback
        if len(matched_entities) < limit:
            remaining = limit - len(matched_entities)
            # Pick top candidate keywords (length >= 3)
            ilike_patterns = [f"%{c}%" for c in candidates[:10] if len(c) >= 3]
            if ilike_patterns:
                sql_ilike = """
                SELECT id, canonical_name, entity_type, display_name, metadata, created_at, updated_at
                FROM entities
                WHERE (canonical_name ILIKE ANY(%s) OR display_name ILIKE ANY(%s))
                  AND id != ALL(%s)
                ORDER BY LENGTH(canonical_name) DESC
                LIMIT %s;
                """
                exclude_ids = list(seen_ids) if seen_ids else [-1]
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql_ilike, (ilike_patterns, ilike_patterns, exclude_ids, remaining))
                    for row in cur.fetchall():
                        eid = row["id"]
                        if eid not in seen_ids:
                            seen_ids.add(eid)
                            matched_entities.append(Entity.from_dict(row))

        return matched_entities

    def _query_authorized_relationships(
        self,
        conn: psycopg.Connection,
        entity_ids: List[int],
        access_context: AccessContext,
        limit: int,
        exclude_rel_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries relationships incident to given entity IDs, strictly applying document
        access control authorization rules via JOIN with documents table.
        """
        if not entity_ids or limit <= 0:
            return []

        allowed_levels = access_context.allowed_access_levels()
        exclude_list = list(exclude_rel_ids) if exclude_rel_ids else [-1]

        sql = """
        SELECT
            r.id,
            r.source_entity_id,
            se.canonical_name AS source_canonical_name,
            se.display_name AS source_display_name,
            se.entity_type AS source_entity_type,
            se.metadata AS source_metadata,
            r.target_entity_id,
            te.canonical_name AS target_canonical_name,
            te.display_name AS target_display_name,
            te.entity_type AS target_entity_type,
            te.metadata AS target_metadata,
            r.relationship_type,
            r.document_id,
            r.chunk_id,
            r.page_number,
            r.metadata AS rel_metadata,
            r.created_at,
            r.updated_at
        FROM relationships r
        JOIN entities se ON r.source_entity_id = se.id
        JOIN entities te ON r.target_entity_id = te.id
        JOIN documents d ON r.document_id = d.id
        WHERE (r.source_entity_id = ANY(%s) OR r.target_entity_id = ANY(%s))
          AND r.id != ALL(%s)
          AND (%s OR d.status = 'active')
          AND (%s OR d.access_level = ANY(%s))
          AND (%s OR d.department = 'public' OR d.department = %s)
        ORDER BY r.id ASC
        LIMIT %s;
        """
        params = (
            entity_ids,
            entity_ids,
            exclude_list,
            access_context.include_archived,
            access_context.is_admin,
            allowed_levels,
            access_context.is_admin,
            access_context.department,
            limit,
        )

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def traverse_graph(
        self,
        conn: psycopg.Connection,
        seed_entities: List[Entity],
        access_context: AccessContext,
        depth: int = 1,
        max_results: int = 50,
        max_neighbors_per_entity: Optional[int] = None,
    ) -> Tuple[List[Relationship], List[Entity]]:
        """
        Traverses PostgreSQL graph outward from seed entities up to specified depth (1 or 2).
        Enforces access control and caps results at max_results with deterministic safety limits.

        Returns:
            Tuple of (traversed_relationships, connected_entities).
        """
        if not seed_entities or max_results <= 0:
            return [], []

        if depth not in (1, 2):
            raise ValueError(f"Traversal depth must be 1 or 2, got {depth}.")

        seed_ids = {e.id for e in seed_entities if e.id is not None}
        seen_rel_ids: Set[int] = set()
        traversed_relationships: List[Relationship] = []
        connected_entities_map: Dict[int, Entity] = {}

        # ----------------------------------------------------------------------
        # Hop 1: Direct relationships incident to seed entities
        # ----------------------------------------------------------------------
        hop1_rows = self._query_authorized_relationships(
            conn=conn,
            entity_ids=list(seed_ids),
            access_context=access_context,
            limit=max_results,
            exclude_rel_ids=seen_rel_ids,
        )

        # Map neighbor_id -> { "entity": Entity, "parent_seed_id": int, "parent_seed_name": str, "parent_edge": Relationship }
        hop1_neighbor_map: Dict[int, Dict[str, Any]] = {}

        for row in hop1_rows:
            rid = row["id"]
            if rid in seen_rel_ids:
                continue
            seen_rel_ids.add(rid)

            # Build Relationship model with provenance
            meta = dict(row.get("rel_metadata") or {})
            s_name = row["source_canonical_name"]
            t_name = row["target_canonical_name"]
            meta["source_name"] = s_name
            meta["target_name"] = t_name
            meta["hop_depth"] = 1
            meta["path"] = f"{s_name} --[{row['relationship_type']}]--> {t_name}"

            rel = Relationship(
                id=rid,
                source_entity_id=row["source_entity_id"],
                target_entity_id=row["target_entity_id"],
                relationship_type=row["relationship_type"],
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                page_number=row["page_number"],
                metadata=meta,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            traversed_relationships.append(rel)

            # Collect neighbor entities connected to seeds
            s_id = row["source_entity_id"]
            t_id = row["target_entity_id"]

            if s_id not in seed_ids:
                if s_id not in connected_entities_map:
                    connected_entities_map[s_id] = Entity(
                        id=s_id,
                        canonical_name=row["source_canonical_name"],
                        entity_type=row["source_entity_type"],
                        display_name=row["source_display_name"],
                        metadata=row.get("source_metadata") or {},
                    )
                if s_id not in hop1_neighbor_map:
                    hop1_neighbor_map[s_id] = {
                        "entity": connected_entities_map[s_id],
                        "parent_seed_id": t_id,
                        "parent_seed_name": t_name,
                        "parent_edge": rel,
                    }

            if t_id not in seed_ids:
                if t_id not in connected_entities_map:
                    connected_entities_map[t_id] = Entity(
                        id=t_id,
                        canonical_name=row["target_canonical_name"],
                        entity_type=row["target_entity_type"],
                        display_name=row["target_display_name"],
                        metadata=row.get("target_metadata") or {},
                    )
                if t_id not in hop1_neighbor_map:
                    hop1_neighbor_map[t_id] = {
                        "entity": connected_entities_map[t_id],
                        "parent_seed_id": s_id,
                        "parent_seed_name": s_name,
                        "parent_edge": rel,
                    }

        # ----------------------------------------------------------------------
        # Hop 2: Traversal from 1-hop neighbors with deterministic safety limits
        # ----------------------------------------------------------------------
        remaining_budget = max_results - len(traversed_relationships)
        if depth == 2 and remaining_budget > 0 and hop1_neighbor_map:
            max_per_entity = max_neighbors_per_entity or self.max_neighbors_per_entity
            hop1_neighbor_ids = list(hop1_neighbor_map.keys())

            # Query candidate relationships incident to hop1 neighbors
            query_batch_limit = min(max(remaining_budget * 2, 50), 100)
            hop2_rows = self._query_authorized_relationships(
                conn=conn,
                entity_ids=hop1_neighbor_ids,
                access_context=access_context,
                limit=query_batch_limit,
                exclude_rel_ids=seen_rel_ids,
            )

            neighbor_edge_counts: Dict[int, int] = {}

            for row in hop2_rows:
                if len(traversed_relationships) >= max_results:
                    break

                rid = row["id"]
                if rid in seen_rel_ids:
                    continue

                s_id = row["source_entity_id"]
                t_id = row["target_entity_id"]

                # Determine which endpoint is the 1-hop neighbor anchor
                if s_id in hop1_neighbor_map:
                    anchor_id = s_id
                    other_id = t_id
                    other_name = row["target_canonical_name"]
                    other_type = row["target_entity_type"]
                    other_display = row["target_display_name"]
                    other_meta = row.get("target_metadata") or {}
                elif t_id in hop1_neighbor_map:
                    anchor_id = t_id
                    other_id = s_id
                    other_name = row["source_canonical_name"]
                    other_type = row["source_entity_type"]
                    other_display = row["source_display_name"]
                    other_meta = row.get("source_metadata") or {}
                else:
                    continue

                parent_info = hop1_neighbor_map[anchor_id]
                parent_seed_id = parent_info["parent_seed_id"]
                parent_seed_name = parent_info["parent_seed_name"]
                parent_edge = parent_info["parent_edge"]

                # Safety Limit 1: Exclude returning immediately to previous entity or any seed (cycle elimination)
                if other_id == parent_seed_id or other_id in seed_ids:
                    continue

                # Safety Limit 2: Maximum neighbors per entity (prevent hub explosion)
                cur_count = neighbor_edge_counts.get(anchor_id, 0)
                if cur_count >= max_per_entity:
                    continue

                neighbor_edge_counts[anchor_id] = cur_count + 1
                seen_rel_ids.add(rid)

                # Path-aware provenance
                meta = dict(row.get("rel_metadata") or {})
                s_name = row["source_canonical_name"]
                t_name = row["target_canonical_name"]
                meta["source_name"] = s_name
                meta["target_name"] = t_name
                meta["hop_depth"] = 2
                meta["parent_edge_id"] = parent_edge.id
                meta["parent_seed_id"] = parent_seed_id
                meta["seed_name"] = parent_seed_name
                meta["hop1_edge"] = {
                    "source": parent_edge.metadata.get("source_name", ""),
                    "relationship_type": parent_edge.relationship_type,
                    "target": parent_edge.metadata.get("target_name", ""),
                }
                meta["hop2_edge"] = {
                    "source": s_name,
                    "relationship_type": row["relationship_type"],
                    "target": t_name,
                }
                e1_path = parent_edge.metadata.get("path", "")
                meta["path"] = f"{e1_path} -> {row['relationship_type']} -> {other_name}"

                rel = Relationship(
                    id=rid,
                    source_entity_id=s_id,
                    target_entity_id=t_id,
                    relationship_type=row["relationship_type"],
                    document_id=row["document_id"],
                    chunk_id=row["chunk_id"],
                    page_number=row["page_number"],
                    metadata=meta,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                traversed_relationships.append(rel)

                if other_id not in seed_ids and other_id not in connected_entities_map:
                    connected_entities_map[other_id] = Entity(
                        id=other_id,
                        canonical_name=other_name,
                        entity_type=other_type,
                        display_name=other_display,
                        metadata=other_meta,
                    )

        # Sort connected entities for deterministic output
        connected_entities = sorted(connected_entities_map.values(), key=lambda e: (e.id or 0))
        return traversed_relationships, connected_entities

    def fetch_chunk_details(
        self,
        conn: psycopg.Connection,
        chunk_ids: List[int],
        access_context: AccessContext,
    ) -> List[Dict[str, Any]]:
        """
        Loads full chunk records and document metadata for authorized chunk IDs.
        """
        if not chunk_ids:
            return []

        allowed_levels = access_context.allowed_access_levels()
        sql = """
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.page_number,
            c.section,
            c.chunk_index,
            c.content,
            d.filename AS document_name,
            d.document_type,
            d.department,
            d.access_level,
            d.status
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.id = ANY(%s)
          AND (%s OR d.status = 'active')
          AND (%s OR d.access_level = ANY(%s))
          AND (%s OR d.department = 'public' OR d.department = %s)
        ORDER BY c.id ASC;
        """
        params = (
            chunk_ids,
            access_context.include_archived,
            access_context.is_admin,
            allowed_levels,
            access_context.is_admin,
            access_context.department,
        )

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def retrieve(
        self,
        query: str,
        access_context: Optional[AccessContext] = None,
        depth: Optional[int] = None,
        max_results: Optional[int] = None,
        max_seed_entities: Optional[int] = None,
        include_chunk_content: bool = False,
        conn: Optional[psycopg.Connection] = None,
        entity_linking_mode: Optional[str] = None,
        traversal_mode: Optional[str] = None,
    ) -> GraphRetrievalResult:
        """
        Executes bounded, access-controlled graph retrieval for a query.

        Args:
            query: User search query or question.
            access_context: Caller authorization context (default: public access).
            depth: Traversal depth (1 or 2). Defaults to self.default_depth or based on traversal_mode.
            max_results: Max relationships to return. Defaults to self.default_max_results.
            max_seed_entities: Max seed entities to discover. Defaults to self.default_max_seeds.
            include_chunk_content: If True, fetches chunk text content for provenance chunks.
            conn: Optional PostgreSQL connection.
            entity_linking_mode: Mode for entity linking ('baseline' or 'improved').
            traversal_mode: Mode for graph traversal ('depth_1' or 'depth_2').

        Returns:
            Structured GraphRetrievalResult containing matched seeds, relationships,
            connected entities, and source provenance IDs.
        """
        start_time = time.perf_counter()

        # Parameter normalization & validation
        clean_query = query.strip() if query else ""
        ctx = access_context or AccessContext()

        t_mode = (traversal_mode or self.traversal_mode).strip().lower()
        if depth is not None:
            traversal_depth = depth
        else:
            traversal_depth = 2 if t_mode == "depth_2" else self.default_depth

        limit = max_results if max_results is not None else self.default_max_results
        seed_limit = max_seed_entities if max_seed_entities is not None else self.default_max_seeds

        if traversal_depth not in (1, 2):
            raise ValueError(
                f"Traversal depth must be 1 or 2 to prevent uncontrolled graph explosion. "
                f"Got depth={traversal_depth}."
            )
        if limit <= 0:
            raise ValueError(f"max_results must be positive, got {limit}.")

        mode = entity_linking_mode or self.entity_linking_mode

        # Handle empty query gracefully
        if not clean_query:
            return GraphRetrievalResult(
                retrieval_metadata={
                    "query": query,
                    "depth": traversal_depth,
                    "max_results": limit,
                    "seed_count": 0,
                    "relationship_count": 0,
                    "connected_entity_count": 0,
                    "source_chunk_count": 0,
                    "document_count": 0,
                    "execution_time_ms": 0.0,
                    "access_context": ctx.to_dict(),
                    "status": "empty_query",
                    "entity_linking_mode": mode,
                    "entity_linking_method": "none",
                    "candidate_entities": [],
                    "linking_execution_time_ms": 0.0,
                }
            )

        should_close = False
        if conn is None:
            conn = get_connection()
            should_close = True

        try:
            # 1. Match seed entities based on configured entity linking mode
            linking_meta: Dict[str, Any] = {}
            if mode == "improved":
                linker = self.get_entity_linker()
                seed_entities, linking_meta = linker.link_entities(
                    conn=conn,
                    query=clean_query,
                    access_context=ctx,
                    limit=seed_limit,
                )
            else:
                seed_entities = self.find_seed_entities(
                    conn=conn,
                    query=clean_query,
                    limit=seed_limit,
                )
                linking_meta = {
                    "entity_linking_method": "baseline_keyword",
                    "candidate_count": len(seed_entities),
                    "authorized_count": len(seed_entities),
                    "candidates": [{"name": e.canonical_name, "method": "baseline_keyword"} for e in seed_entities],
                    "execution_time_ms": 0.0,
                }

            # If no seed entities match, return clean empty result
            if not seed_entities:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return GraphRetrievalResult(
                    retrieval_metadata={
                        "query": clean_query,
                        "depth": traversal_depth,
                        "max_results": limit,
                        "seed_count": 0,
                        "relationship_count": 0,
                        "connected_entity_count": 0,
                        "source_chunk_count": 0,
                        "document_count": 0,
                        "execution_time_ms": elapsed_ms,
                        "access_context": ctx.to_dict(),
                        "status": "no_seed_entities_found",
                        "entity_linking_mode": mode,
                        "entity_linking_method": linking_meta.get("entity_linking_method", "none"),
                        "candidate_entities": linking_meta.get("candidates", []),
                        "linking_execution_time_ms": linking_meta.get("execution_time_ms", 0.0),
                    }
                )

            # 2. Traverse graph up to specified depth with access control
            relationships, connected_entities = self.traverse_graph(
                conn=conn,
                seed_entities=seed_entities,
                access_context=ctx,
                depth=traversal_depth,
                max_results=limit,
                max_neighbors_per_entity=self.max_neighbors_per_entity,
            )

            # 3. Collect and deduplicate provenance IDs from authorized relationships
            chunk_ids_set: Set[int] = set()
            doc_ids_set: Set[int] = set()
            page_numbers_set: Set[int] = set()

            for rel in relationships:
                if rel.chunk_id is not None:
                    chunk_ids_set.add(rel.chunk_id)
                if rel.document_id is not None:
                    doc_ids_set.add(rel.document_id)
                if rel.page_number is not None:
                    page_numbers_set.add(rel.page_number)

            source_chunk_ids = sorted(chunk_ids_set)
            document_ids = sorted(doc_ids_set)
            page_numbers = sorted(page_numbers_set)

            # 4. Optional chunk content fetch
            source_chunks: List[Dict[str, Any]] = []
            if include_chunk_content and source_chunk_ids:
                source_chunks = self.fetch_chunk_details(
                    conn=conn,
                    chunk_ids=source_chunk_ids,
                    access_context=ctx,
                )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            retrieval_metadata = {
                "query": clean_query,
                "depth": traversal_depth,
                "traversal_mode": "depth_2" if traversal_depth == 2 else "depth_1",
                "max_results": limit,
                "seed_count": len(seed_entities),
                "relationship_count": len(relationships),
                "connected_entity_count": len(connected_entities),
                "source_chunk_count": len(source_chunk_ids),
                "document_count": len(document_ids),
                "execution_time_ms": elapsed_ms,
                "access_context": ctx.to_dict(),
                "status": "success",
                "entity_linking_mode": mode,
                "entity_linking_method": linking_meta.get("entity_linking_method", "unknown"),
                "candidate_entities": linking_meta.get("candidates", []),
                "linking_execution_time_ms": linking_meta.get("execution_time_ms", 0.0),
            }

            return GraphRetrievalResult(
                matched_entities=seed_entities,
                relationships=relationships,
                connected_entities=connected_entities,
                source_chunk_ids=source_chunk_ids,
                document_ids=document_ids,
                page_numbers=page_numbers,
                retrieval_metadata=retrieval_metadata,
                source_chunks=source_chunks,
            )

        finally:
            if should_close and conn is not None:
                conn.close()
