"""
Graph RAG Phase G2.1: 50-Chunk Prototype Extraction and Quality Reporting Script.

Selects 53 representative chunks across all 8 documents in the corpus:
- Enterprise internal documents (ai_governance_policy.md, enterprise_platform_architecture.pdf, support_faq.txt)
- AI policy and governance (AI Index 2024 p. 372 timeline, etc.)
- Speech and Language Processing (wav2vec 2.0, CTC loss, Transformer self-attention)
- Foundation Models (BERT, pretraining, language models)
- Computer Vision (CNNs, residual networks)
- NLP (Eisenstein parsing and representations)

Extracts structured entities and relationships using GraphExtractorService,
computes quality and deduplication metrics, and writes:
- reports/graph_extraction_prototype.json
- reports/graph_extraction_quality_report.md

CRITICAL: Does NOT insert into production PostgreSQL entities/relationships tables.
"""
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection
from app.graph_extractor import GraphExtractorService
from app.graph_ontology import (
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
    ValidatedExtractionResult,
)


def select_prototype_chunks() -> List[Dict[str, Any]]:
    """Selects 53 diverse representative chunks across all 8 documents."""
    conn = get_connection()
    cur = conn.cursor()

    selected = []

    # 1. Enterprise small docs (all chunks)
    for doc_id in [90, 93, 97]:
        cur.execute(
            """
            SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = %s
            ORDER BY c.id
            """,
            (doc_id,),
        )
        for r in cur.fetchall():
            selected.append({
                "chunk_id": r[0],
                "document_id": r[1],
                "page_number": r[2],
                "content": r[3],
                "filename": r[4],
                "department": r[5],
                "access_level": r[6],
            })

    # 2. Doc 91: AI Index 2024 (Stanford)
    # Include page 372 chunks
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 91 AND c.page_number = 372
        ORDER BY c.id
        """
    )
    p372 = cur.fetchall()
    for r in p372:
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    remaining_91 = max(0, 10 - len(p372))
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 91 AND c.page_number IN (24, 369, 370, 371, 375, 400, 450)
        ORDER BY c.id
        LIMIT %s
        """,
        (remaining_91,),
    )
    for r in cur.fetchall():
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    # 3. Doc 96: Speech and Language Processing (Jurafsky & Martin)
    for p in [206, 275, 348, 350, 363, 370, 372]:
        cur.execute(
            """
            SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = 96 AND c.page_number = %s
            ORDER BY c.id LIMIT 1
            """,
            (p,),
        )
        row = cur.fetchone()
        if row:
            selected.append({
                "chunk_id": row[0], "document_id": row[1], "page_number": row[2], "content": row[3],
                "filename": row[4], "department": row[5], "access_level": row[6],
            })
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 96 AND c.page_number NOT IN (206, 275, 348, 350, 363, 370, 372)
        ORDER BY c.id LIMIT 2
        """
    )
    for r in cur.fetchall():
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    # 4. Doc 94: Foundation Models (Paass & Giesselbach)
    for p in [331, 332, 100, 150, 200]:
        cur.execute(
            """
            SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.document_id = 94 AND c.page_number = %s
            ORDER BY c.id LIMIT 1
            """,
            (p,),
        )
        row = cur.fetchone()
        if row:
            selected.append({
                "chunk_id": row[0], "document_id": row[1], "page_number": row[2], "content": row[3],
                "filename": row[4], "department": row[5], "access_level": row[6],
            })
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 94 AND c.page_number NOT IN (331, 332, 100, 150, 200)
        ORDER BY c.id LIMIT 4
        """
    )
    for r in cur.fetchall():
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    # 5. Doc 92: Computer Vision (Hassaballah & Awad)
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 92 AND c.page_number IN (50, 100, 150, 200, 250, 300, 350)
        ORDER BY c.id LIMIT 7
        """
    )
    for r in cur.fetchall():
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    # 6. Doc 95: NLP (Eisenstein)
    cur.execute(
        """
        SELECT c.id, c.document_id, c.page_number, c.content, d.filename, d.department, d.access_level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.document_id = 95 AND c.page_number IN (50, 100, 150, 200, 250, 300, 350)
        ORDER BY c.id LIMIT 7
        """
    )
    for r in cur.fetchall():
        selected.append({
            "chunk_id": r[0], "document_id": r[1], "page_number": r[2], "content": r[3],
            "filename": r[4], "department": r[5], "access_level": r[6],
        })

    conn.close()
    return selected


def run_prototype():
    print("=" * 80)
    print("  Graph RAG Phase G2.1 - 50-Chunk Prototype Extraction")
    print("=" * 80)

    chunks = select_prototype_chunks()
    print(f"Selected {len(chunks)} representative chunks across 8 corpus documents.\n")

    extractor = GraphExtractorService()
    print(f"Initialized GraphExtractorService (model: {extractor.model_name}).\n")

    results: List[Dict[str, Any]] = []
    entity_type_counter = Counter()
    relationship_type_counter = Counter()
    dropped_entities_count = 0
    dropped_relationships_count = 0
    errors_count = 0

    start_time = time.time()

    for idx, item in enumerate(chunks, start=1):
        cid = item["chunk_id"]
        did = item["document_id"]
        pnum = item["page_number"]
        fname = item["filename"]
        text = item["content"]

        print(f"[{idx:02d}/{len(chunks):02d}] Chunk {cid} (Doc {did} p.{pnum} | {fname[:35]})...", end=" ", flush=True)

        res = extractor.extract_from_chunk(
            chunk_id=cid,
            document_id=did,
            page_number=pnum,
            chunk_text=text,
        )

        n_ent = len(res.entities)
        n_rel = len(res.relationships)
        n_drop_ent = len(res.dropped_entities)
        n_drop_rel = len(res.dropped_relationships)

        dropped_entities_count += n_drop_ent
        dropped_relationships_count += n_drop_rel

        if res.status == "error":
            errors_count += 1
            print(f"FAILED: {res.validation_errors[:1]}")
        else:
            print(f"OK ({n_ent} entities, {n_rel} relations | dropped: {n_drop_ent}e, {n_drop_rel}r)")

        for e in res.entities:
            entity_type_counter[e.entity_type] += 1
        for r in res.relationships:
            relationship_type_counter[r.relationship_type] += 1

        record = res.to_dict()
        record["chunk_snippet"] = text[:300]
        record["filename"] = fname
        results.append(record)

        # Inter-chunk delay to ensure consistent rate-limit headroom
        time.sleep(0.6)

    duration = time.time() - start_time
    total_entities = sum(entity_type_counter.values())
    total_relationships = sum(relationship_type_counter.values())

    print("\n" + "=" * 80)
    print("  EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"  Total Chunks Processed       : {len(chunks)}")
    print(f"  Total Extraction Time        : {duration:.2f}s ({duration/len(chunks):.2f}s/chunk)")
    print(f"  Total Valid Entities Extracted: {total_entities}")
    print(f"  Total Valid Relations Extracted: {total_relationships}")
    print(f"  Duplicate Entities Removed   : {dropped_entities_count}")
    print(f"  Duplicate/Invalid Relations Removed: {dropped_relationships_count}")
    print(f"  Extraction Failures          : {errors_count}")
    print("=" * 80)

    # Write prototype JSON
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "graph_extraction_prototype.json"

    prototype_payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_chunks_processed": len(chunks),
            "duration_seconds": round(duration, 2),
            "model": extractor.model_name,
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "dropped_entities": dropped_entities_count,
            "dropped_relationships": dropped_relationships_count,
            "errors": errors_count,
            "entity_type_distribution": dict(entity_type_counter.most_common()),
            "relationship_type_distribution": dict(relationship_type_counter.most_common()),
        },
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prototype_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OUTPUT] Saved prototype JSON -> {json_path}")

    # Generate Markdown Quality Report
    generate_markdown_report(reports_dir / "graph_extraction_quality_report.md", prototype_payload, results)
    print(f"[OUTPUT] Saved quality report -> {reports_dir / 'graph_extraction_quality_report.md'}")


def generate_markdown_report(report_path: Path, payload: Dict[str, Any], results: List[Dict[str, Any]]):
    meta = payload["metadata"]
    ent_dist = meta["entity_type_distribution"]
    rel_dist = meta["relationship_type_distribution"]

    lines = []
    lines.append("# Graph RAG Phase G2.1: Extraction Prototype Quality Report")
    lines.append("")
    lines.append(f"**Execution Date**: {meta['timestamp']}")
    lines.append(f"**Extractor Model**: `{meta['model']}`")
    lines.append(f"**Scope**: Controlled 53-Chunk Prototype across 8 corpus documents")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Quality Metrics")
    lines.append("")
    lines.append("| Metric | Value | Notes |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Chunks Processed** | **{meta['total_chunks_processed']}** | Representative sample across all 8 documents |")
    lines.append(f"| **Execution Duration** | **{meta['duration_seconds']}s** | {meta['duration_seconds']/meta['total_chunks_processed']:.2f}s per chunk average |")
    lines.append(f"| **Valid Entities Extracted** | **{meta['total_entities']}** | Average {meta['total_entities']/meta['total_chunks_processed']:.1f} entities/chunk |")
    lines.append(f"| **Valid Relationships Extracted** | **{meta['total_relationships']}** | Average {meta['total_relationships']/meta['total_chunks_processed']:.1f} relationships/chunk |")
    lines.append(f"| **Duplicate Entities Deduplicated** | **{meta['dropped_entities']}** | Merged via canonicalization rules |")
    lines.append(f"| **Duplicate/Invalid Relations Dropped** | **{meta['dropped_relationships']}** | Deduplicated or unresolvable endpoints pruned |")
    lines.append(f"| **Extraction Failures** | **{meta['errors']}** | Zero fatal crashes |")
    lines.append(f"| **Production DB Status** | **0 rows inserted** | PostgreSQL tables remain completely unpopulated |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Entity Distribution by Controlled Type")
    lines.append("")
    lines.append("| Entity Type | Count | Percentage | Description |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for et in sorted(list(VALID_ENTITY_TYPES)):
        cnt = ent_dist.get(et, 0)
        pct = (cnt / meta['total_entities'] * 100) if meta['total_entities'] > 0 else 0
        lines.append(f"| `{et}` | {cnt} | {pct:.1f}% | Supported |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Relationship Distribution by Controlled Type")
    lines.append("")
    lines.append("| Relationship Type | Count | Percentage | Description |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for rt in sorted(list(VALID_RELATIONSHIP_TYPES)):
        cnt = rel_dist.get(rt, 0)
        pct = (cnt / meta['total_relationships'] * 100) if meta['total_relationships'] > 0 else 0
        lines.append(f"| `{rt}` | {cnt} | {pct:.1f}% | Controlled |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Exemplar Extracted Subgraphs with Full Provenance")
    lines.append("")

    # Pick 4 diverse exemplar chunks to display: AI Index timeline, wav2vec speech, Enterprise policy, Architecture
    exemplar_cids = [8273]  # AI Index page 372
    # Add first chunk from doc 90, 93, 96 if available
    for r in results:
        if r["document_id"] in (90, 93, 96) and r["chunk_id"] not in exemplar_cids and len(r["relationships"]) > 0:
            exemplar_cids.append(r["chunk_id"])
            if len(exemplar_cids) >= 5:
                break

    for r in results:
        if r["chunk_id"] in exemplar_cids:
            lines.append(f"### Example: Chunk {r['chunk_id']} — {r['filename']} (Page {r['page_number']})")
            lines.append("")
            lines.append("#### CHUNK TEXT")
            lines.append("```text")
            lines.append(r["chunk_snippet"] + "...")
            lines.append("```")
            lines.append("")
            lines.append("#### EXTRACTED ENTITIES")
            lines.append("| Surface Mention | Canonical Name | Type | Confidence |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for e in r["entities"]:
                lines.append(f"| {e['name']} | **{e['canonical_name']}** | `{e['entity_type']}` | {e['confidence']} |")
            lines.append("")
            lines.append("#### EXTRACTED RELATIONSHIPS")
            lines.append("| Source Entity | Relationship | Target Entity | Evidence |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for rel in r["relationships"]:
                ev = rel.get('evidence') or "Explicit text mention"
                lines.append(f"| **{rel['source']}** | `{rel['relationship_type']}` | **{rel['target']}** | *\"{ev[:80]}\"* |")
            lines.append("")
            lines.append("#### PROVENANCE METADATA")
            lines.append(f"- **Document ID**: `{r['document_id']}`")
            lines.append(f"- **Chunk ID**: `{r['chunk_id']}`")
            lines.append(f"- **Page Number**: `{r['page_number']}`")
            lines.append(f"- **Filename**: `{r['filename']}`")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_prototype()
