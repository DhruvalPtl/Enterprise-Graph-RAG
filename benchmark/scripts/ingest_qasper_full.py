"""
Ingestion script for the full-fidelity Attention Is All You Need paper.
"""
import sys
import time
import json
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection, insert_document, insert_chunks, insert_chunk_graph_transaction
from app.chunker import RecursiveStructuralChunker
from app.embeddings import EmbeddingService
from app.graph_extractor import GraphExtractorService
from app.graph_ontology import ValidatedExtractionResult
from app.models import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qasper_full_ingest")


def run():
    paper_path = BASE_DIR / "benchmark" / "datasets" / "02_qasper" / "qasper_paper.txt"
    filename = "qasper_attention_paper.txt"
    text = paper_path.read_text(encoding="utf-8")
    logger.info(f"Loaded full paper text: {len(text)} characters, {len(text.splitlines())} lines.")

    # 1. Clean up any existing records for this document in DB
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
            existing = cur.fetchall()
            for row in existing:
                doc_id = row[0]
                cur.execute("DELETE FROM relationships WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            conn.commit()
            logger.info("Cleared previous instances of qasper_attention_paper.txt from DB.")

    # 2. Insert new document record
    with get_connection() as conn:
        doc_id = insert_document(
            conn=conn,
            filename=filename,
            document_type="scientific_paper",
            department="engineering",
            access_level="employee",
            status="active",
        )
        logger.info(f"Created new document record ID: {doc_id}")

    # 3. Chunk with RecursiveStructuralChunker
    chunker = RecursiveStructuralChunker(chunk_size=800, chunk_overlap=150)
    doc_obj = Document(id=filename, content=text, metadata={"filename": filename})
    raw_chunks = chunker.chunk_document(doc_obj)
    logger.info(f"Chunked document into {len(raw_chunks)} chunks.")

    # 4. Generate embeddings
    embedder = EmbeddingService()
    chunk_texts = [c.text for c in raw_chunks]
    embeddings = embedder.embed_texts(chunk_texts)
    logger.info(f"Generated {len(embeddings)} embeddings locally.")

    # 5. Insert chunks
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
            cur.execute("SELECT id, chunk_index FROM chunks WHERE document_id = %s ORDER BY chunk_index ASC", (doc_id,))
            inserted_chunks = cur.fetchall()
            inserted_chunk_ids = [r[0] for r in inserted_chunks]
        logger.info(f"Saved {len(inserted_chunk_ids)} chunks into PostgreSQL.")

    # 6. Extract Knowledge Graph via Gemini (15 RPM safe)
    extractor = GraphExtractorService()
    total_entities = 0
    total_relationships = 0

    logger.info(f"Beginning Knowledge Graph extraction for {len(inserted_chunk_ids)} chunks...")
    for idx, (cid, chunk_text) in enumerate(zip(inserted_chunk_ids, chunk_texts), start=1):
        logger.info(f"Extracting graph chunk [{idx}/{len(inserted_chunk_ids)}] (Chunk ID {cid})...")
        try:
            extraction_result: ValidatedExtractionResult = extractor.extract_from_chunk(
                chunk_id=cid,
                document_id=doc_id,
                chunk_text=chunk_text,
            )
            with get_connection() as conn:
                ents_upserted, rels_inserted = insert_chunk_graph_transaction(conn, extraction_result)
                total_entities += ents_upserted
                total_relationships += rels_inserted
            logger.info(f"  -> Extracted {ents_upserted} entities, {rels_inserted} relationships.")
        except Exception as e:
            logger.warning(f"  -> Extraction warning on chunk {cid}: {e}")

        if idx < len(inserted_chunk_ids):
            time.sleep(4.5)

    # 7. Update manifests and benchmark dataset with new chunk IDs
    manifest_path = BASE_DIR / "benchmark" / "datasets" / "ingestion_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["qasper"] = {
            "document_id": doc_id,
            "filename": filename,
            "chunk_ids": inserted_chunk_ids,
            "total_chunks": len(inserted_chunk_ids),
            "total_entities": total_entities,
            "total_relationships": total_relationships,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    # Update benchmark.json corpus document ID
    benchmark_json_path = BASE_DIR / "benchmark" / "datasets" / "02_qasper" / "benchmark.json"
    if benchmark_json_path.exists():
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        bdata["corpus_documents"] = [{
            "document_id": doc_id,
            "filename": filename,
            "department": "engineering",
            "access_level": "employee",
        }]
        # Map question expected source chunks dynamically based on chunk contents
        for q in bdata.get("questions", []):
            qid = q.get("id")
            matching_cids = []
            if qid == "QASPER-01":  # d_model = 512
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "d_model = 512" in ctxt]
            elif qid == "QASPER-02":  # h = 8, d_k = d_v = 64
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "h = 8" in ctxt and "64" in ctxt]
            elif qid == "QASPER-03":  # BLEU 28.4
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "28.4" in ctxt]
            elif qid == "QASPER-04":  # decoder masking
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "mask" in ctxt.lower()]
            elif qid == "QASPER-05":  # formula
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "Attention(Q, K, V)" in ctxt]
            elif qid == "QASPER-06":  # hardware 8 P100
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "P100" in ctxt and "100,000 steps" in ctxt]
            elif qid == "QASPER-07":  # positional encoding
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "PE(pos" in ctxt]
            elif qid == "QASPER-08":  # d_ff = 4096 vs 2048
                matching_cids = [cid for cid, ctxt in zip(inserted_chunk_ids, chunk_texts) if "4096" in ctxt]
            if matching_cids:
                q["expected_source_chunks"] = matching_cids

        with open(benchmark_json_path, "w", encoding="utf-8") as f:
            json.dump(bdata, f, indent=2)

    logger.info(f"Successfully finished ingestion of full paper! Doc ID: {doc_id}, Chunks: {len(inserted_chunk_ids)}, Entities: {total_entities}, Relationships: {total_relationships}")


if __name__ == "__main__":
    run()
