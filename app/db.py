"""
PostgreSQL Database and Vector Storage Layer for Enterprise RAG (Step 3).

This module manages relational persistence and vector storage for documents and chunks
using psycopg 3 and pgvector.

Tables:
- documents: Tracks ingested source documents, filenames, types, and content hashes.
- chunks: Stores textual chunk passages and 384-dimensional MiniLM embeddings
  linked to parent documents via foreign key (ON DELETE CASCADE).

Indexes:
- idx_chunks_document_id: B-tree index for fast document-level chunk retrieval.
- idx_chunks_embedding_hnsw: HNSW index using cosine distance (vector_cosine_ops)
  for sub-millisecond approximate nearest neighbor vector search.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import psycopg
from psycopg.rows import dict_row

from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    DATABASE_URL,
    ALL_CHUNKS_FILE,
    EMBEDDED_CHUNKS_FILE,
    VECTOR_DIMENSION,
)

# Relational and Vector DDL for documents and chunks (extended in Step 9)
SCHEMA_SQL = f"""
-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents metadata table with access control attributes (Step 9)
CREATE TABLE IF NOT EXISTS documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename TEXT NOT NULL,
    document_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_hash TEXT,
    department TEXT NOT NULL DEFAULT 'public',
    access_level TEXT NOT NULL DEFAULT 'public',
    status TEXT NOT NULL DEFAULT 'active'
);

-- Idempotent schema migrations for existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS department TEXT NOT NULL DEFAULT 'public';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS access_level TEXT NOT NULL DEFAULT 'public';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_documents_access ON documents(department, access_level, status);

-- 3. Chunks table with 384-dimensional vector column (MiniLM)
CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR({VECTOR_DIMENSION}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Backward compatibility migration for pre-existing chunks tables
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding VECTOR({VECTOR_DIMENSION});

-- 4. B-tree index for relational joins & cascade queries
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- 5. HNSW index for high-recall, sub-millisecond cosine vector similarity search
-- Uses vector_cosine_ops to match cosine distance operator (<=>).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- 6. Entities table (Graph RAG Phase G1)
CREATE TABLE IF NOT EXISTS entities (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_entities_canonical_type UNIQUE (canonical_name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_canonical_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

-- 7. Relationships table (Graph RAG Phase G1)
CREATE TABLE IF NOT EXISTS relationships (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
    page_number INTEGER,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_relationships_edge_provenance UNIQUE (source_entity_id, target_entity_id, relationship_type, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_document_id ON relationships(document_id);
CREATE INDEX IF NOT EXISTS idx_relationships_chunk_id ON relationships(chunk_id);
"""



def get_db_config() -> Dict[str, Any]:
    """Returns the database connection parameters from environment/config."""
    if DATABASE_URL:
        return {"conninfo": DATABASE_URL}
    return {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "dbname": POSTGRES_DB,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "connect_timeout": 3,
    }


def get_connection(autocommit: bool = False, connect_timeout: Optional[int] = None) -> psycopg.Connection:
    """
    Creates and returns a psycopg connection to the PostgreSQL database.
    Registers pgvector types on the connection if available.
    Raises ConnectionError with beginner-friendly guidance if connection fails.
    """
    config = dict(get_db_config())
    if connect_timeout is not None:
        config["connect_timeout"] = connect_timeout

    try:
        if "conninfo" in config:
            conn = psycopg.connect(config["conninfo"], autocommit=autocommit)
        else:
            conn = psycopg.connect(**config, autocommit=autocommit)

        # Register pgvector type handler on connection
        try:
            from pgvector.psycopg import register_vector
            register_vector(conn)
        except Exception:
            pass

        return conn
    except Exception as e:
        safe_host = config.get("host", "localhost")
        safe_port = config.get("port", 5432)
        safe_db = config.get("dbname", "rag_db")
        raise ConnectionError(
            f"Could not connect to PostgreSQL on {safe_host}:{safe_port}/{safe_db}. "
            f"Please verify that your PostgreSQL service is running and credentials in .env are correct. "
            f"Underlying error: {type(e).__name__}"
        ) from e


def test_connection(timeout: int = 1) -> bool:
    """Tests whether PostgreSQL is reachable and returns True/False rapidly."""
    try:
        with get_connection(connect_timeout=timeout) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return cur.fetchone() is not None
    except Exception:
        return False


def init_db(conn: Optional[psycopg.Connection] = None) -> None:
    """
    Executes the DDL schema to enable vector extension, create tables, and create indexes.
    Safe to run repeatedly (uses IF NOT EXISTS clauses).
    """
    should_close = False
    if conn is None:
        conn = get_connection(autocommit=True)
        should_close = True

    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        if not conn.autocommit:
            conn.commit()
    finally:
        if should_close:
            conn.close()


def _sanitize_pg_text(val: Optional[Any]) -> Optional[str]:
    """Strips PostgreSQL-incompatible NUL (0x00) bytes from string inputs."""
    if val is None:
        return None
    return str(val).replace("\x00", "")


def insert_document(
    conn: psycopg.Connection,
    filename: str,
    document_type: Optional[str] = None,
    content_hash: Optional[str] = None,
    department: str = "public",
    access_level: str = "public",
    status: str = "active",
) -> int:
    """
    Inserts a new document record with access control metadata (Step 9)
    and returns the generated BIGINT primary key ID.
    """
    clean_filename = _sanitize_pg_text(filename) or "unnamed_document"
    clean_doc_type = _sanitize_pg_text(document_type)
    clean_hash = _sanitize_pg_text(content_hash)
    clean_dept = _sanitize_pg_text(department) or "public"
    clean_level = _sanitize_pg_text(access_level) or "public"
    clean_status = _sanitize_pg_text(status) or "active"

    sql = """
    INSERT INTO documents (filename, document_type, content_hash, department, access_level, status)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (clean_filename, clean_doc_type, clean_hash, clean_dept, clean_level, clean_status))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to insert document: no ID returned.")
        return row[0]


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: List[Dict[str, Any]],
) -> int:
    """
    Inserts a list of chunk dictionaries linked to a parent document_id.
    Validates that embedding vectors match VECTOR_DIMENSION (384) when provided,
    explicitly rejecting incompatible dimensions (e.g., 768-d Gemini embeddings).
    Returns the count of inserted chunks.
    """
    if not chunks:
        return 0

    try:
        from pgvector.psycopg import register_vector
        register_vector(conn)
    except Exception:
        pass

    sql = """
    INSERT INTO chunks (document_id, content, page_number, section, chunk_index, embedding)
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    params = []
    for chunk in chunks:
        emb = chunk.get("embedding")
        if emb is not None:
            emb_len = len(emb)
            if emb_len != VECTOR_DIMENSION:
                raise ValueError(
                    f"Incompatible embedding dimension: expected {VECTOR_DIMENSION} (MiniLM), "
                    f"got {emb_len}. Non-{VECTOR_DIMENSION} vectors (such as 768-d Gemini embeddings) "
                    f"cannot be stored in the {VECTOR_DIMENSION}-d pgvector index."
                )

        raw_content = chunk.get("text", chunk.get("content", "")) or ""
        clean_content = _sanitize_pg_text(raw_content)

        sec = chunk.get("metadata", {}).get("section", chunk.get("section"))
        clean_sec = _sanitize_pg_text(sec)

        page_num = chunk.get("metadata", {}).get("page_number", chunk.get("page_number"))
        chunk_idx = chunk.get("metadata", {}).get("chunk_index", chunk.get("chunk_index", 0))

        params.append((
            document_id,
            clean_content,
            page_num,
            clean_sec,
            chunk_idx,
            emb,
        ))

    with conn.cursor() as cur:
        cur.executemany(sql, params)
    return len(params)


def store_processed_chunks(
    chunks_file: Optional[Path] = None,
    conn: Optional[psycopg.Connection] = None,
) -> Dict[str, Any]:
    """
    Reads embedded chunks (defaults to EMBEDDED_CHUNKS_FILE if present, else ALL_CHUNKS_FILE),
    groups chunks by document, and persists documents and chunks (including embeddings)
    into PostgreSQL.
    Preserves all existing JSON files.
    """
    if chunks_file is None:
        chunks_file = EMBEDDED_CHUNKS_FILE if EMBEDDED_CHUNKS_FILE.exists() else ALL_CHUNKS_FILE
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data: List[Dict[str, Any]] = json.load(f)

    if not chunks_data:
        return {"documents_stored": 0, "chunks_stored": 0}

    # Group chunks by document_id
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks_data:
        doc_key = c.get("document_id", "default_doc")
        grouped.setdefault(doc_key, []).append(c)

    should_close = False
    if conn is None:
        conn = get_connection(autocommit=False)
        should_close = True

    documents_count = 0
    chunks_count = 0

    try:
        for doc_key, doc_chunks in grouped.items():
            first_chunk = doc_chunks[0]
            meta = first_chunk.get("metadata", {})
            filename = meta.get("document_name", f"{doc_key}.txt")
            doc_type = meta.get("file_type", Path(filename).suffix.lstrip("."))
            content_hash = doc_key.replace("doc_", "")

            # Map demonstration access metadata (Step 9)
            demo_access = {
                "support_faq.txt": {"department": "public", "access_level": "public", "status": "active"},
                "enterprise_platform_architecture.pdf": {"department": "engineering", "access_level": "employee", "status": "active"},
                "ai_governance_policy.md": {"department": "engineering", "access_level": "manager", "status": "active"},
            }.get(filename, {})

            doc_dept = meta.get("department", demo_access.get("department", "public"))
            doc_level = meta.get("access_level", demo_access.get("access_level", "public"))
            doc_stat = meta.get("status", demo_access.get("status", "active"))

            # 1. Insert parent document row with access control attributes
            doc_id = insert_document(
                conn,
                filename=filename,
                document_type=doc_type,
                content_hash=content_hash,
                department=doc_dept,
                access_level=doc_level,
                status=doc_stat,
            )
            documents_count += 1

            # 2. Insert referencing chunk rows with embeddings
            inserted = insert_chunks(conn, document_id=doc_id, chunks=doc_chunks)
            chunks_count += inserted

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()

    return {
        "documents_stored": documents_count,
        "chunks_stored": chunks_count,
    }


# ==============================================================================
# Graph RAG Phase G1: Entity & Relationship Database Operations
# ==============================================================================

def insert_entity(
    conn: psycopg.Connection,
    canonical_name: str,
    entity_type: str,
    display_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Inserts a canonical entity into the entities table (Phase G1).
    If an entity with the same (canonical_name, entity_type) already exists,
    updates metadata/display_name and returns the existing entity's ID, ensuring deduplication.
    """
    clean_canonical = _sanitize_pg_text(canonical_name).strip() if canonical_name else ""
    clean_type = _sanitize_pg_text(entity_type).strip() if entity_type else ""
    clean_display = _sanitize_pg_text(display_name).strip() if display_name else ""
    meta_json = json.dumps(metadata or {})

    if not clean_canonical or not clean_type:
        raise ValueError("Both canonical_name and entity_type are required.")

    sql = """
    INSERT INTO entities (canonical_name, entity_type, display_name, metadata)
    VALUES (%s, %s, %s, %s::jsonb)
    ON CONFLICT (canonical_name, entity_type) DO UPDATE
        SET display_name = EXCLUDED.display_name,
            metadata = entities.metadata || EXCLUDED.metadata,
            updated_at = NOW()
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (clean_canonical, clean_type, clean_display, meta_json))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to insert or resolve entity ID.")
        return row[0]


def get_entity(
    conn: psycopg.Connection,
    entity_id: int,
) -> Optional[Dict[str, Any]]:
    """Retrieves an entity record by its primary key ID."""
    sql = """
    SELECT id, canonical_name, entity_type, display_name, metadata, created_at, updated_at
    FROM entities
    WHERE id = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (entity_id,))
        return cur.fetchone()


def get_entity_by_canonical_name(
    conn: psycopg.Connection,
    canonical_name: str,
    entity_type: str,
) -> Optional[Dict[str, Any]]:
    """Retrieves an entity record by canonical_name and entity_type."""
    clean_canonical = canonical_name.strip() if canonical_name else ""
    clean_type = entity_type.strip() if entity_type else ""
    sql = """
    SELECT id, canonical_name, entity_type, display_name, metadata, created_at, updated_at
    FROM entities
    WHERE canonical_name = %s AND entity_type = %s;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (clean_canonical, clean_type))
        return cur.fetchone()


def insert_relationship(
    conn: psycopg.Connection,
    source_entity_id: int,
    target_entity_id: int,
    relationship_type: str,
    document_id: int,
    chunk_id: Optional[int] = None,
    page_number: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Inserts a directed semantic relationship edge with source material provenance (Phase G1).
    Validates foreign key integrity through PostgreSQL constraints.
    Prevents duplicate insertions for the same (source, target, rel_type, chunk_id) context.
    """
    clean_type = _sanitize_pg_text(relationship_type).strip() if relationship_type else ""
    if not clean_type:
        raise ValueError("relationship_type is required.")
    meta_json = json.dumps(metadata or {})

    sql = """
    INSERT INTO relationships (
        source_entity_id,
        target_entity_id,
        relationship_type,
        document_id,
        chunk_id,
        page_number,
        metadata
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
    ON CONFLICT (source_entity_id, target_entity_id, relationship_type, chunk_id) DO UPDATE
        SET metadata = relationships.metadata || EXCLUDED.metadata,
            updated_at = NOW()
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            source_entity_id,
            target_entity_id,
            clean_type,
            document_id,
            chunk_id,
            page_number,
            meta_json,
        ))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to insert or resolve relationship ID.")
        return row[0]


def get_relationships_for_entity(
    conn: psycopg.Connection,
    entity_id: int,
    direction: str = "both",
) -> List[Dict[str, Any]]:
    """
    Retrieves relationships connected to an entity ID.
    direction: 'outgoing', 'incoming', or 'both'.
    """
    if direction == "outgoing":
        condition = "r.source_entity_id = %s"
        params = (entity_id,)
    elif direction == "incoming":
        condition = "r.target_entity_id = %s"
        params = (entity_id,)
    else:
        condition = "(r.source_entity_id = %s OR r.target_entity_id = %s)"
        params = (entity_id, entity_id)

    sql = f"""
    SELECT
        r.id,
        r.source_entity_id,
        se.canonical_name AS source_canonical_name,
        se.display_name AS source_display_name,
        se.entity_type AS source_entity_type,
        r.target_entity_id,
        te.canonical_name AS target_canonical_name,
        te.display_name AS target_display_name,
        te.entity_type AS target_entity_type,
        r.relationship_type,
        r.document_id,
        r.chunk_id,
        r.page_number,
        r.metadata,
        r.created_at,
        r.updated_at
    FROM relationships r
    JOIN entities se ON r.source_entity_id = se.id
    JOIN entities te ON r.target_entity_id = te.id
    WHERE {condition}
    ORDER BY r.id ASC;
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def insert_chunk_graph_transaction(
    conn: psycopg.Connection,
    result: Any,
) -> Tuple[int, int]:
    """
    Persists all extracted entities and relationships for a single chunk within an atomic transaction (Phase G2.2).
    If an error occurs, the transaction automatically rolls back, guaranteeing zero partial graph data.

    Args:
        conn: Active psycopg connection.
        result: ValidatedExtractionResult instance.

    Returns:
        Tuple of (entities_upserted, relationships_inserted).
    """
    if not getattr(result, "entities", None) and not getattr(result, "relationships", None):
        return 0, 0

    with conn.transaction():
        # 1. Upsert entities and build canonical lookup map
        canonical_to_id: Dict[str, int] = {}
        for ent in result.entities:
            c_name = ent.canonical_name or ent.name
            e_type = ent.entity_type
            d_name = ent.name
            meta = {
                "description": getattr(ent, "description", None),
                "confidence": getattr(ent, "confidence", 1.0),
            }
            ent_id = insert_entity(
                conn,
                canonical_name=c_name,
                entity_type=e_type,
                display_name=d_name,
                metadata=meta,
            )
            canonical_to_id[c_name.strip().lower()] = ent_id
            canonical_to_id[ent.name.strip().lower()] = ent_id

        # 2. Insert relationships with resolved entity IDs
        rels_inserted = 0
        for rel in result.relationships:
            src_id = canonical_to_id.get(rel.source.strip().lower())
            tgt_id = canonical_to_id.get(rel.target.strip().lower())

            if src_id is None or tgt_id is None:
                continue

            rel_meta = {
                "evidence": getattr(rel, "evidence", None),
                "confidence": getattr(rel, "confidence", 1.0),
            }
            insert_relationship(
                conn,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=rel.relationship_type,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                page_number=result.page_number,
                metadata=rel_meta,
            )
            rels_inserted += 1

    conn.commit()
    return len(result.entities), rels_inserted


# ==============================================================================
# Document Management & Inspection Helpers
# ==============================================================================

def get_loaded_documents(conn: Optional[psycopg.Connection] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all documents currently stored in PostgreSQL with aggregated chunk counts.
    """
    should_close = False
    if conn is None:
        conn = get_connection(autocommit=True)
        should_close = True

    sql = """
    SELECT 
        d.id,
        d.filename,
        d.document_type,
        d.department,
        d.access_level,
        d.status,
        d.created_at,
        COUNT(DISTINCT c.id) AS chunk_count,
        COUNT(DISTINCT r.id) AS relationship_count
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    LEFT JOIN relationships r ON d.id = r.document_id
    GROUP BY d.id, d.filename, d.document_type, d.department, d.access_level, d.status, d.created_at
    ORDER BY d.id DESC;
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        if should_close:
            conn.close()


def delete_document(document_id: int, conn: Optional[psycopg.Connection] = None) -> bool:
    """
    Deletes a document by ID. Cascades automatically to chunks and relationships.
    """
    should_close = False
    if conn is None:
        conn = get_connection(autocommit=False)
        should_close = True

    sql = "DELETE FROM documents WHERE id = %s RETURNING id;"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (document_id,))
            deleted = cur.fetchone() is not None
        if not conn.autocommit:
            conn.commit()
        return deleted
    except Exception:
        if not conn.autocommit:
            conn.rollback()
        raise
    finally:
        if should_close:
            conn.close()


def get_document_chunks(document_id: int, conn: Optional[psycopg.Connection] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all text chunks belonging to a document ordered by chunk_index.
    """
    should_close = False
    if conn is None:
        conn = get_connection(autocommit=True)
        should_close = True

    sql = """
    SELECT id, chunk_index, page_number, section, content
    FROM chunks
    WHERE document_id = %s
    ORDER BY chunk_index ASC;
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (document_id,))
            return cur.fetchall()
    finally:
        if should_close:
            conn.close()


def get_document_relationships(document_id: int, conn: Optional[psycopg.Connection] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all knowledge graph relationships extracted from a specific document.
    """
    should_close = False
    if conn is None:
        conn = get_connection(autocommit=True)
        should_close = True

    sql = """
    SELECT
        r.id,
        se.canonical_name AS source_name,
        se.entity_type AS source_type,
        r.relationship_type,
        te.canonical_name AS target_name,
        te.entity_type AS target_type,
        r.chunk_id,
        r.page_number
    FROM relationships r
    JOIN entities se ON r.source_entity_id = se.id
    JOIN entities te ON r.target_entity_id = te.id
    WHERE r.document_id = %s
    ORDER BY r.id ASC;
    """
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, (document_id,))
            return cur.fetchall()
    finally:
        if should_close:
            conn.close()



