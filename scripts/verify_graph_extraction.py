"""
Graph RAG Phase G2.2: Post-Extraction Validation and Quality Audit Script.

Performs:
1. Database referential and relational integrity verification.
2. Provenance resolution checks (relationship -> chunk -> document).
3. Graph topological statistics (degrees, entity distributions, relation distributions).
4. Qualitative sampling of 50 extracted subgraphs across all 8 corpus documents.
5. Saves audit report to reports/graph_extraction_quality_audit_50_samples.md.
"""
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection
from app.graph_ontology import VALID_ENTITY_TYPES, VALID_RELATIONSHIP_TYPES


def verify_database_integrity() -> Dict[str, Any]:
    print("=" * 80)
    print("  GRAPH RAG PHASE G2.2: DATABASE INTEGRITY VERIFICATION")
    print("=" * 80)

    conn = get_connection()
    results = {}

    try:
        with conn.cursor() as cur:
            # 1. Row counts
            cur.execute("SELECT count(*) FROM documents;")
            doc_cnt = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks;")
            chk_cnt = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM entities;")
            ent_cnt = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM relationships;")
            rel_cnt = cur.fetchone()[0]

            print(f"  Documents Count        : {doc_cnt} (Expected: 8)")
            print(f"  Chunks Count           : {chk_cnt:,} (Expected: 11,609)")
            print(f"  Entities Count         : {ent_cnt:,}")
            print(f"  Relationships Count    : {rel_cnt:,}")
            print("-" * 80)

            results["documents"] = doc_cnt
            results["chunks"] = chk_cnt
            results["entities"] = ent_cnt
            results["relationships"] = rel_cnt

            # 2. Orphan check
            cur.execute(
                """
                SELECT count(*) FROM relationships r
                WHERE NOT EXISTS (SELECT 1 FROM entities WHERE id = r.source_entity_id)
                   OR NOT EXISTS (SELECT 1 FROM entities WHERE id = r.target_entity_id);
                """
            )
            orphan_entities = cur.fetchone()[0]
            print(f"  Orphan Entity Edges    : {orphan_entities} (PASS = 0)")

            cur.execute(
                """
                SELECT count(*) FROM relationships r
                WHERE NOT EXISTS (SELECT 1 FROM chunks WHERE id = r.chunk_id)
                   OR NOT EXISTS (SELECT 1 FROM documents WHERE id = r.document_id);
                """
            )
            orphan_provenance = cur.fetchone()[0]
            print(f"  Orphan Provenance Edges: {orphan_provenance} (PASS = 0)")

            # 3. Self loops
            cur.execute("SELECT count(*) FROM relationships WHERE source_entity_id = target_entity_id;")
            self_loops = cur.fetchone()[0]
            print(f"  Self-Referential Loops : {self_loops} (PASS = 0)")

            # 4. Invalid types
            valid_ents = list(VALID_ENTITY_TYPES)
            cur.execute("SELECT count(*) FROM entities WHERE NOT (entity_type = ANY(%s));", (valid_ents,))
            invalid_ent_types = cur.fetchone()[0]
            print(f"  Invalid Entity Types   : {invalid_ent_types} (PASS = 0)")

            valid_rels = list(VALID_RELATIONSHIP_TYPES)
            cur.execute("SELECT count(*) FROM relationships WHERE NOT (relationship_type = ANY(%s));", (valid_rels,))
            invalid_rel_types = cur.fetchone()[0]
            print(f"  Invalid Relation Types : {invalid_rel_types} (PASS = 0)")

            # 5. NULL provenance checks
            cur.execute("SELECT count(*) FROM relationships WHERE chunk_id IS NULL OR document_id IS NULL;")
            null_prov = cur.fetchone()[0]
            print(f"  NULL Provenance Links  : {null_prov} (PASS = 0)")

            print("=" * 80)

            all_passed = (
                doc_cnt == 8
                and chk_cnt == 11609
                and orphan_entities == 0
                and orphan_provenance == 0
                and self_loops == 0
                and invalid_ent_types == 0
                and invalid_rel_types == 0
                and null_prov == 0
            )
            print(f"  DATABASE INTEGRITY STATUS: {'ALL CHECKS PASSED' if all_passed else 'FAILURES DETECTED'}\n")
            results["integrity_passed"] = all_passed
            return results

    finally:
        conn.close()


def generate_graph_topography() -> Dict[str, Any]:
    print("=" * 80)
    print("  GRAPH TOPOGRAPHICAL METRICS")
    print("=" * 80)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM entities;")
            ent_cnt = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM relationships;")
            rel_cnt = cur.fetchone()[0]

            avg_deg = (rel_cnt * 2 / max(1, ent_cnt))

            print(f"  Average Degree per Entity: {avg_deg:.2f}")

            # Top 10 entity types
            cur.execute("SELECT entity_type, count(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC LIMIT 10;")
            print("\n  Top Entity Types:")
            for r in cur.fetchall():
                print(f"    - {r[0]:<15}: {r[1]:,}")

            # Top 10 relationship types
            cur.execute("SELECT relationship_type, count(*) as cnt FROM relationships GROUP BY relationship_type ORDER BY cnt DESC LIMIT 10;")
            print("\n  Top Relationship Types:")
            for r in cur.fetchall():
                print(f"    - {r[0]:<15}: {r[1]:,}")

            # Top 10 connected hub entities
            cur.execute(
                """
                SELECT se.canonical_name, se.entity_type, count(*) as degree
                FROM (
                    SELECT source_entity_id as ent_id FROM relationships
                    UNION ALL
                    SELECT target_entity_id as ent_id FROM relationships
                ) all_edges
                JOIN entities se ON all_edges.ent_id = se.id
                GROUP BY se.id, se.canonical_name, se.entity_type
                ORDER BY degree DESC
                LIMIT 10;
                """
            )
            print("\n  Top 10 Connected Hub Entities:")
            for r in cur.fetchall():
                print(f"    - {r[0]} ({r[1]}): {r[2]:,} connections")

            # Document edge counts
            cur.execute(
                """
                SELECT d.filename, count(r.id) as edges
                FROM documents d
                LEFT JOIN relationships r ON d.id = r.document_id
                GROUP BY d.id, d.filename
                ORDER BY edges DESC;
                """
            )
            print("\n  Extracted Relationships by Document:")
            for r in cur.fetchall():
                print(f"    - {r[0][:40]:<40}: {r[1]:,} edges")

            print("=" * 80 + "\n")
            return {"avg_degree": avg_deg}
    finally:
        conn.close()


def generate_quality_sample_audit(sample_size: int = 50):
    print("=" * 80)
    print(f"  GENERATING {sample_size}-CHUNK QUALITY AUDIT REPORT")
    print("=" * 80)

    conn = get_connection()
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "graph_extraction_quality_audit_50_samples.md"

    try:
        with conn.cursor() as cur:
            # Select chunk IDs that have at least one extracted relationship
            cur.execute(
                """
                SELECT DISTINCT chunk_id
                FROM relationships
                ORDER BY chunk_id ASC;
                """
            )
            available_chunks = [r[0] for r in cur.fetchall()]

            if not available_chunks:
                print("No relationships found in database to sample.")
                return

            random.seed(42)
            sampled_ids = random.sample(available_chunks, min(sample_size, len(available_chunks)))

            lines = [
                f"# Graph RAG Phase G2.2: {len(sampled_ids)}-Chunk Quality Audit Report",
                "",
                f"**Date**: {Path(__file__).stat().st_mtime}",
                f"**Sample Size**: {len(sampled_ids)} chunks randomly selected across documents with extracted graphs",
                "",
                "---",
                "",
            ]

            for idx, cid in enumerate(sampled_ids, start=1):
                cur.execute(
                    """
                    SELECT c.id, c.document_id, c.page_number, c.content, d.filename
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = %s;
                    """,
                    (cid,),
                )
                chunk_row = cur.fetchone()
                if not chunk_row:
                    continue

                chunk_id, doc_id, page_num, content, filename = chunk_row

                # Fetch relationships and connected entities for this chunk
                cur.execute(
                    """
                    SELECT se.canonical_name, se.entity_type, r.relationship_type, te.canonical_name, te.entity_type, r.metadata
                    FROM relationships r
                    JOIN entities se ON r.source_entity_id = se.id
                    JOIN entities te ON r.target_entity_id = te.id
                    WHERE r.chunk_id = %s
                    ORDER BY r.id ASC;
                    """,
                    (cid,),
                )
                rels = cur.fetchall()

                # Collect unique entities in this chunk
                chunk_entities = {}
                for r in rels:
                    chunk_entities[r[0]] = r[1]
                    chunk_entities[r[3]] = r[4]

                lines.append(f"## Sample {idx}/{len(sampled_ids)}: Chunk {chunk_id} — {filename} (Page {page_num})")
                lines.append("")
                lines.append("### 1. Source Chunk Text")
                lines.append("```text")
                snippet = content.strip()[:400] + ("..." if len(content.strip()) > 400 else "")
                lines.append(snippet)
                lines.append("```")
                lines.append("")
                lines.append("### 2. Extracted Entities")
                lines.append("| Canonical Entity Name | Type |")
                lines.append("| :--- | :--- |")
                for ename, etype in sorted(chunk_entities.items()):
                    lines.append(f"| **{ename}** | `{etype}` |")
                lines.append("")
                lines.append("### 3. Extracted Relationships")
                lines.append("| Source Entity | Relationship | Target Entity | Evidence / Metadata |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for r in rels:
                    meta = r[5] or {}
                    raw_ev = meta.get("evidence") or "Explicit text mention"
                    evidence = str(raw_ev)[:70].replace("\n", " ")
                    lines.append(f"| **{r[0]}** | `{r[2]}` | **{r[3]}** | *\"{evidence}\"* |")
                lines.append("")
                lines.append("### 4. Provenance Metadata")
                lines.append(f"- **Document ID**: `{doc_id}`")
                lines.append(f"- **Chunk ID**: `{chunk_id}`")
                lines.append(f"- **Page Number**: `{page_num}`")
                lines.append(f"- **Filename**: `{filename}`")
                lines.append("")
                lines.append("---")
                lines.append("")

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Saved quality audit report to: {report_path}\n")

    finally:
        conn.close()


def main():
    verify_database_integrity()
    generate_graph_topography()
    generate_quality_sample_audit(sample_size=50)


if __name__ == "__main__":
    main()
