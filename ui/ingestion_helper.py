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
    get_document_chunks,
    insert_chunk_graph_transaction,
)


def ingest_document_file(
    file_bytes: bytes,
    filename: str,
    department: str = "public",
    access_level: str = "public",
    extract_graph: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Ingests an uploaded document into the Enterprise Graph RAG database.

    Args:
        file_bytes: Raw binary content of the uploaded document
        filename: Document file name (e.g. 'quarterly_report.pdf')
        department: Access department ('public', 'engineering', 'finance', 'hr')
        access_level: Access clearance ('public', 'employee', 'manager', 'admin')
        extract_graph: If True and Gemini API key is available, extracts graph triples
        progress_callback: Optional callback(current, total) for UI progress updates

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
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
    if extract_graph:
        if not api_key:
            graph_stats["error"] = "Gemini API key is required for Knowledge Graph extraction. Please enter your key in the sidebar Control Panel."
        else:
            try:
                from app.graph_extractor import GraphExtractorService
                extractor = GraphExtractorService(api_key=api_key)
                total_ents = 0
                total_rels = 0

                # Fetch real database chunks with primary keys (chunks.id)
                db_chunks = get_document_chunks(doc_id)
                total_db_chunks = len(db_chunks)

                with get_connection() as conn:
                    for idx, c_info in enumerate(db_chunks):
                        if progress_callback:
                            progress_callback(idx + 1, total_db_chunks)

                        real_chunk_id = c_info["id"]
                        chunk_text = c_info.get("content", "")
                        page_num = c_info.get("page_number")

                        res = extractor.extract_from_chunk(
                            chunk_id=real_chunk_id,
                            document_id=doc_id,
                            chunk_text=chunk_text,
                            page_number=page_num,
                        )
                        if res and getattr(res, "status", None) not in ("skipped_empty", "error"):
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

