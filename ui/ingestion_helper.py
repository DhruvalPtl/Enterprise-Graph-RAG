"""
Document Ingestion Helper for Streamlit UI.

Provides an end-to-end ingestion pipeline callable from the UI:
1. Saves uploaded raw file (PDF, Markdown, TXT) to data/raw/.
2. Breaks document into structured chunks with section/page awareness.
3. Computes 384-dimensional dense vectors using MiniLM (sentence-transformers).
4. Persists document and chunk vectors into PostgreSQL pgvector with ACID safety.
5. Optionally extracts Knowledge Graph entities and relationship triples via Gemini.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

from app.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTOR_DIMENSION,
    GEMINI_API_KEY,
)
from app.pipeline import IngestionPipeline
from app.embeddings import EmbeddingService
from app.db import (
    get_connection,
    init_db,
    insert_document,
    insert_chunks,
    insert_chunk_graph_transaction,
)


def ingest_document_file(
    file_bytes: bytes,
    filename: str,
    department: str = "public",
    access_level: str = "public",
    extract_graph: bool = False,
) -> Dict[str, Any]:
    """
    Ingests an uploaded document into the Enterprise Graph RAG database.

    Args:
        file_bytes: Raw binary content of the uploaded document
        filename: Document file name (e.g. 'quarterly_report.pdf')
        department: Access department ('public', 'engineering', 'finance', 'hr')
        access_level: Access clearance ('public', 'employee', 'manager', 'admin')
        extract_graph: If True and Gemini API key is available, extracts graph triples

    Returns:
        Dict with execution summary (doc_id, chunks_count, vector_dim, graph_stats, etc.)
    """
    # 1. Ensure target directory exists and save raw file
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_file_path = RAW_DATA_DIR / filename
    with open(raw_file_path, "wb") as f:
        f.write(file_bytes)

    # 2. Structure-aware chunking
    pipeline = IngestionPipeline(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)
    chunks = pipeline.process_file(raw_file_path)

    if not chunks:
        raise ValueError(f"No extractable text or chunks could be generated from '{filename}'.")

    # 3. Generate dense embeddings (MiniLM 384-d for PostgreSQL pgvector)
    embedder = EmbeddingService(provider="sentence-transformers")
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # 4. Prepare chunk dicts with embeddings
    chunk_dicts = []
    for idx, (c, emb) in enumerate(zip(chunks, embeddings)):
        meta = c.metadata if hasattr(c, "metadata") and isinstance(c.metadata, dict) else {}
        chunk_dicts.append({
            "text": c.text,
            "page_number": meta.get("page_number") or getattr(c, "page_number", None),
            "section": meta.get("section") or getattr(c, "section", None),
            "chunk_index": idx,
            "embedding": emb,
        })

    # 5. Persist to PostgreSQL
    with get_connection() as conn:
        init_db(conn)

        # File extension as doc type
        doc_type = Path(filename).suffix.lstrip(".").lower() or "txt"

        doc_id = insert_document(
            conn=conn,
            filename=filename,
            document_type=doc_type,
            department=department,
            access_level=access_level,
            status="active",
        )

        chunks_stored = insert_chunks(
            conn=conn,
            document_id=doc_id,
            chunks=chunk_dicts,
        )
        conn.commit()

    # 6. Optional Knowledge Graph Extraction
    graph_stats = {"entities_upserted": 0, "relationships_inserted": 0, "extracted": False}
    if extract_graph and GEMINI_API_KEY:
        try:
            from app.graph_extractor import GraphExtractorService
            extractor = GraphExtractorService()
            total_ents = 0
            total_rels = 0

            with get_connection() as conn:
                for idx, c_dict in enumerate(chunk_dicts):
                    res = extractor.extract(
                        text=c_dict["text"],
                        chunk_id=idx + 1,
                        document_id=doc_id,
                        page_number=c_dict.get("page_number"),
                    )
                    if res:
                        ents, rels = insert_chunk_graph_transaction(conn, res)
                        total_ents += ents
                        total_rels += rels
                conn.commit()

            graph_stats = {
                "entities_upserted": total_ents,
                "relationships_inserted": total_rels,
                "extracted": True,
            }
        except Exception as e:
            graph_stats["error"] = str(e)

    return {
        "success": True,
        "filename": filename,
        "document_id": doc_id,
        "chunks_stored": chunks_stored,
        "vector_dimension": VECTOR_DIMENSION,
        "department": department,
        "access_level": access_level,
        "graph_stats": graph_stats,
    }


def preload_sample_documents(extract_graph: bool = False) -> List[Dict[str, Any]]:
    """
    Ingests the bundled sample documents from data/raw/ into PostgreSQL.
    """
    sample_configs = [
        ("support_faq.txt", "public", "public"),
        ("enterprise_platform_architecture.pdf", "engineering", "employee"),
        ("ai_governance_policy.md", "engineering", "manager"),
    ]
    results = []
    for filename, dept, clearance in sample_configs:
        file_path = RAW_DATA_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                bytes_data = f.read()
            res = ingest_document_file(
                file_bytes=bytes_data,
                filename=filename,
                department=dept,
                access_level=clearance,
                extract_graph=extract_graph,
            )
            results.append(res)
    return results

