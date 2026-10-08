"""
Ingests authentic MuSiQue and GraphRAG-Bench benchmark corpora into PostgreSQL with rate-limited Gemini Graph extraction.
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
logger = logging.getLogger("official_ingest")

CORPORA = [
    {
        "filename": "musique_official_corpus.txt",
        "path": BASE_DIR / "benchmark" / "datasets" / "03_musique_2wiki" / "musique_corpus.txt",
        "bench_json": BASE_DIR / "benchmark" / "datasets" / "03_musique_2wiki" / "benchmark.json",
        "doc_type": "multi_hop_corpus",
        "department": "engineering",
        "access_level": "employee",
        "dataset_key": "musique",
    },
    {
        "filename": "graphrag_bench_official.txt",
        "path": BASE_DIR / "benchmark" / "datasets" / "04_graphrag_bench" / "graphrag_bench_corpus.txt",
        "bench_json": BASE_DIR / "benchmark" / "datasets" / "04_graphrag_bench" / "benchmark.json",
        "doc_type": "medical_graph_network",
        "department": "engineering",
        "access_level": "employee",
        "dataset_key": "graphrag_bench",
    },
]


def ingest_one(item: dict, embedder: EmbeddingService, chunker: RecursiveStructuralChunker, extractor: GraphExtractorService):
    filename = item["filename"]
    filepath = item["path"]
    text = filepath.read_text(encoding="utf-8")
    logger.info(f"=== Starting Ingestion: {filename} ({len(text)} chars) ===")

    # 1. Clean previous record if any
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
            for row in cur.fetchall():
                doc_id = row[0]
                cur.execute("DELETE FROM relationships WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            conn.commit()

    # 2. Insert doc
    with get_connection() as conn:
        doc_id = insert_document(
            conn=conn,
            filename=filename,
            document_type=item["doc_type"],
            department=item["department"],
            access_level=item["access_level"],
            status="active",
        )
        logger.info(f"Created document {filename} ID: {doc_id}")

    # 3. Chunk
    doc_obj = Document(id=filename, content=text, metadata={"filename": filename})
    raw_chunks = chunker.chunk_document(doc_obj)
    chunk_texts = [c.text for c in raw_chunks]
    logger.info(f"Chunked into {len(raw_chunks)} chunks.")

    # 4. Embed
    embeddings = embedder.embed_texts(chunk_texts)
    logger.info(f"Generated {len(embeddings)} local MiniLM embeddings.")

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

    # 6. Extract graph
    total_entities = 0
    total_relationships = 0
    logger.info(f"Extracting Knowledge Graph across {len(inserted_chunk_ids)} chunks...")
    for idx, (cid, chunk_text) in enumerate(zip(inserted_chunk_ids, chunk_texts), start=1):
        logger.info(f"[{item['dataset_key'].upper()}] Graph chunk [{idx}/{len(inserted_chunk_ids)}] (ID {cid})...")
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

    # 7. Update benchmark.json
    bench_path = item["bench_json"]
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            bdata = json.load(f)
        bdata["corpus_documents"] = [{
            "document_id": doc_id,
            "filename": filename,
            "department": item["department"],
            "access_level": item["access_level"],
        }]
        # Automatically map expected chunks
        for q in bdata.get("questions", []):
            q_evs = q.get("evidence", [])
            ent_matches = q.get("expected_entities", [])
            matched_cids = []
            for cid, ctxt in zip(inserted_chunk_ids, chunk_texts):
                for ent in ent_matches:
                    if ent.lower() in ctxt.lower():
                        matched_cids.append(cid)
                        break
                for ev in q_evs:
                    if ev[:30].lower() in ctxt.lower() and cid not in matched_cids:
                        matched_cids.append(cid)
            q["expected_source_chunks"] = matched_cids

        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(bdata, f, indent=2)

    logger.info(f"Finished {filename}: Doc ID {doc_id}, Chunks {len(inserted_chunk_ids)}, Entities {total_entities}, Rels {total_relationships}.")
    return doc_id, inserted_chunk_ids, total_entities, total_relationships


def main():
    embedder = EmbeddingService()
    chunker = RecursiveStructuralChunker(chunk_size=800, chunk_overlap=150)
    extractor = GraphExtractorService()

    manifest_path = BASE_DIR / "benchmark" / "datasets" / "ingestion_manifest.json"
    manifest_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    for item in CORPORA:
        doc_id, chunk_ids, ents, rels = ingest_one(item, embedder, chunker, extractor)
        manifest_data[item["dataset_key"]] = {
            "document_id": doc_id,
            "filename": item["filename"],
            "chunk_ids": chunk_ids,
            "total_chunks": len(chunk_ids),
            "total_entities": ents,
            "total_relationships": rels,
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("All official corpora successfully ingested and manifests updated!")


if __name__ == "__main__":
    main()
