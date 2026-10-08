"""
Safe, Rate-Limited Benchmark Corpus Ingestion Script.

Ingests benchmark corpora into PostgreSQL:
1. QASPER Scientific Attention Paper
2. MuSiQue / 2WikiMultiHop Multi-Hop Reasoning Corpus
3. GraphRAG-Bench Enterprise Dependency Corpus

Strictly respects Google AI Studio Free Tier limits by enforcing a 4.5-second sleep
between chunk extraction calls (staying well under the 15 RPM limit).
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import psycopg
from app.db import get_connection, insert_document, insert_chunks, insert_chunk_graph_transaction
from app.chunker import RecursiveStructuralChunker
from app.embeddings import EmbeddingService
from app.graph_extractor import GraphExtractorService
from app.graph_ontology import ValidatedExtractionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_ingest")

CORPUS_FILES = [
    {
        "filename": "qasper_attention_paper.txt",
        "path": BASE_DIR / "benchmark" / "datasets" / "02_qasper" / "qasper_paper.txt",
        "doc_type": "scientific_paper",
        "department": "engineering",
        "access_level": "employee",
        "dataset_key": "qasper",
    },
    {
        "filename": "musique_multi_hop_corpus.txt",
        "path": BASE_DIR / "benchmark" / "datasets" / "03_musique_2wiki" / "musique_corpus.txt",
        "doc_type": "multi_hop_corpus",
        "department": "engineering",
        "access_level": "employee",
        "dataset_key": "musique",
    },
    {
        "filename": "graphrag_bench_enterprise.txt",
        "path": BASE_DIR / "benchmark" / "datasets" / "04_graphrag_bench" / "graphrag_bench_corpus.txt",
        "doc_type": "enterprise_architecture",
        "department": "engineering",
        "access_level": "employee",
        "dataset_key": "graphrag_bench",
    },
]


def ingest_corpus_file(item: Dict[str, Any], embedder: EmbeddingService, chunker: RecursiveStructuralChunker, extractor: GraphExtractorService) -> Dict[str, Any]:
    filepath = Path(item["path"])
    filename = item["filename"]
    logger.info(f"=== Starting Ingestion for: {filename} ===")

    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8")

    # 1. Check if document already exists in DB
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
            existing = cur.fetchone()
            if existing:
                doc_id = existing[0]
                logger.info(f"Document {filename} already exists with ID {doc_id}. Re-indexing chunks...")
                # Delete existing chunks, entities, relationships for this doc
                cur.execute("DELETE FROM relationships WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                conn.commit()
            else:
                doc_id = insert_document(
                    conn=conn,
                    filename=filename,
                    document_type=item["doc_type"],
                    department=item["department"],
                    access_level=item["access_level"],
                    status="active",
                )
                logger.info(f"Created document record ID: {doc_id}")

    # 2. Chunk text
    from app.models import Document
    doc_obj = Document(id=filename, content=text, metadata={"filename": filename})
    raw_chunks = chunker.chunk_document(doc_obj)
    logger.info(f"Chunked into {len(raw_chunks)} chunks.")

    # 3. Generate embeddings
    chunk_texts = [c.text for c in raw_chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    logger.info(f"Generated {len(embeddings)} embeddings locally via MiniLM.")

    # 4. Insert chunks into DB
    chunk_rows = []
    for idx, (c, emb) in enumerate(zip(raw_chunks, embeddings)):
        meta = c.metadata or {}
        chunk_rows.append({
            "document_id": doc_id,
            "chunk_index": idx,
            "content": c.text,
            "embedding": emb,
            "page_number": meta.get("page_number", 1),
            "section": meta.get("section", "General"),
            "metadata": {"char_count": len(c.text), "filename": filename},
        })

    with get_connection() as conn:
        insert_chunks(conn=conn, document_id=doc_id, chunks=chunk_rows)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM chunks WHERE document_id = %s ORDER BY chunk_index ASC", (doc_id,))
            inserted_chunk_ids = [r[0] for r in cur.fetchall()]
        logger.info(f"Saved {len(inserted_chunk_ids)} chunks into PostgreSQL.")

    # 5. Extract Knowledge Graph Entities & Relationships via Gemini (Rate-Limited)
    total_entities_extracted = 0
    total_relationships_extracted = 0

    logger.info(f"Beginning LLM graph extraction across {len(inserted_chunk_ids)} chunks...")
    for idx, (cid, chunk_text) in enumerate(zip(inserted_chunk_ids, chunk_texts), start=1):
        logger.info(f"Extracting graph from chunk [{idx}/{len(inserted_chunk_ids)}] (Chunk ID {cid})...")
        try:
            extraction_result: ValidatedExtractionResult = extractor.extract_from_chunk(
                chunk_id=cid,
                document_id=doc_id,
                chunk_text=chunk_text,
            )
            # Save entities & relationships atomically with foreign key integrity
            with get_connection() as conn:
                ents_upserted, rels_inserted = insert_chunk_graph_transaction(conn, extraction_result)
                total_entities_extracted += ents_upserted
                total_relationships_extracted += rels_inserted
            logger.info(f"  -> Extracted {ents_upserted} entities, {rels_inserted} relationships.")
        except Exception as e:
            logger.warning(f"  -> Extraction warning on chunk {cid}: {e}")

        # Rate-limiting sleep to strictly observe 15 RPM free limit
        if idx < len(inserted_chunk_ids):
            time.sleep(4.5)

    logger.info(f"Ingestion complete for {filename}: {len(inserted_chunk_ids)} chunks, {total_entities_extracted} entities, {total_relationships_extracted} relationships.")
    return {
        "document_id": doc_id,
        "filename": filename,
        "chunk_ids": inserted_chunk_ids,
        "chunk_texts": chunk_texts,
        "total_chunks": len(inserted_chunk_ids),
        "total_entities": total_entities_extracted,
        "total_relationships": total_relationships_extracted,
    }


def main():
    logger.info("Initializing embedding model, chunker, and graph extractor...")
    embedder = EmbeddingService()
    chunker = RecursiveStructuralChunker(chunk_size=800, chunk_overlap=150)
    extractor = GraphExtractorService()

    ingestion_manifest = {}
    for item in CORPUS_FILES:
        res = ingest_corpus_file(item, embedder, chunker, extractor)
        ingestion_manifest[item["dataset_key"]] = res

    # Save ingestion manifest with chunk IDs for gold dataset generation
    manifest_path = BASE_DIR / "benchmark" / "datasets" / "ingestion_manifest.json"
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        # Save without chunk_texts to keep compact
        clean_manifest = {
            k: {
                "document_id": v["document_id"],
                "filename": v["filename"],
                "chunk_ids": v["chunk_ids"],
                "total_chunks": v["total_chunks"],
                "total_entities": v["total_entities"],
                "total_relationships": v["total_relationships"],
            }
            for k, v in ingestion_manifest.items()
        }
        json.dump(clean_manifest, f, indent=2)

    logger.info(f"All benchmark corpora successfully ingested! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
