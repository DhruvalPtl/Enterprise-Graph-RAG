"""
Corpus Cleanup Script for Fast Graph RAG Testing.

1. Moves original PDF files of the 4 archived books to data/archived_books/.
2. Moves processed/chunked JSON files to data/archived_books/processed/.
3. Filters all_chunks.json and embedded_chunks.json to only contain active book chunks.
4. Removes non-active documents and chunks from PostgreSQL (cascading chunks and relationships).
5. Cleans up orphan graph entities without active relationships.
6. Removes non-active chunk rows from SQLite checkpoint tracking.
7. Verifies and outputs the active corpus state.
"""
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVED_DIR = DATA_DIR / "archived_books"
ARCHIVED_PROCESSED_DIR = ARCHIVED_DIR / "processed"
CHECKPOINT_DB_PATH = DATA_DIR / "graph_extraction_progress.db"

# The two books to KEEP
KEPT_BOOK_FILENAMES = {
    "Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf",
    "speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf",
}

# The four books to ARCHIVE
ARCHIVED_BOOK_FILENAMES = [
    "deep-learning-in-computer-vision-principles-and-applications-mahmoud-hassaballah-and-ali-ismail-awad-841.pdf",
    "foundation-models-for-natural-language-processing-gerhard-paa-and-sven-giesselbach-839.pdf",
    "natural-language-processing-jacob-eisenstein-838.pdf",
    "enterprise_platform_architecture.pdf",
]

# Additional non-book files to archive
ADDITIONAL_ARCHIVED_FILES = [
    "ai_governance_policy.md",
    "support_faq.txt",
]


def main():
    print("=" * 80)
    print("  CORPUS CLEANUP FOR FAST GRAPH RAG TESTING")
    print("=" * 80)

    # 1. Create Archive Directories
    ARCHIVED_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVED_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1] Created archive directories:\n    - {ARCHIVED_DIR}\n    - {ARCHIVED_PROCESSED_DIR}\n")

    # 2. Move original PDF files of the four archived books
    print("[2] Moving original PDF files to data/archived_books/...")
    moved_raw_files = []
    for fname in ARCHIVED_BOOK_FILENAMES:
        src = RAW_DIR / fname
        dst = ARCHIVED_DIR / fname
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_raw_files.append(fname)
            print(f"    -> Moved: {fname} -> data/archived_books/")
        elif dst.exists():
            print(f"    -> Already in archive: {fname}")
            moved_raw_files.append(fname)
        else:
            print(f"    -> Warning: File not found: {src}")

    # Also archive copies of additional non-book files if desired
    for fname in ADDITIONAL_ARCHIVED_FILES:
        src = RAW_DIR / fname
        dst = ARCHIVED_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"    -> Archived copy: {fname} -> data/archived_books/")

    print(f"    Total original PDF files archived: {len(moved_raw_files)}\n")

    # 3. Move processed chunk files to archive
    print("[3] Moving processed/chunked JSON files to data/archived_books/processed/...")
    chunk_file_prefixes = [
        "deep-learning-in-computer-vision-principles-and-applications-mahmoud-hassaballah-and-ali-ismail-awad-841",
        "foundation-models-for-natural-language-processing-gerhard-paa-and-sven-giesselbach-839",
        "natural-language-processing-jacob-eisenstein-838",
        "enterprise_platform_architecture",
        "ai_governance_policy",
        "support_faq",
    ]
    moved_processed = []
    for f in PROCESSED_DIR.glob("*_chunks.json"):
        for prefix in chunk_file_prefixes:
            if f.name.startswith(prefix):
                dst = ARCHIVED_PROCESSED_DIR / f.name
                shutil.move(str(f), str(dst))
                moved_processed.append(f.name)
                print(f"    -> Moved chunk file: {f.name}")
                break

    print(f"    Total processed chunk files archived: {len(moved_processed)}\n")

    # 4. Filter all_chunks.json and embedded_chunks.json
    print("[4] Filtering all_chunks.json and embedded_chunks.json...")
    all_chunks_path = PROCESSED_DIR / "all_chunks.json"
    embedded_chunks_path = PROCESSED_DIR / "embedded_chunks.json"

    if all_chunks_path.exists():
        # Backup full version
        shutil.copy2(str(all_chunks_path), str(ARCHIVED_PROCESSED_DIR / "all_chunks_full_11609.json"))
        with open(all_chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        filtered_all = [
            c for c in all_chunks
            if (c.get("document_name") in KEPT_BOOK_FILENAMES or
                c.get("metadata", {}).get("document_name") in KEPT_BOOK_FILENAMES)
        ]
        with open(all_chunks_path, "w", encoding="utf-8") as f:
            json.dump(filtered_all, f, indent=2)
        print(f"    -> all_chunks.json: {len(all_chunks):,} -> {len(filtered_all):,} chunks")

    if embedded_chunks_path.exists():
        # Backup full version
        shutil.copy2(str(embedded_chunks_path), str(ARCHIVED_PROCESSED_DIR / "embedded_chunks_full_11609.json"))
        with open(embedded_chunks_path, "r", encoding="utf-8") as f:
            embedded_chunks = json.load(f)
        filtered_embedded = [
            c for c in embedded_chunks
            if (c.get("document_name") in KEPT_BOOK_FILENAMES or
                c.get("metadata", {}).get("document_name") in KEPT_BOOK_FILENAMES)
        ]
        with open(embedded_chunks_path, "w", encoding="utf-8") as f:
            json.dump(filtered_embedded, f, indent=2)
        print(f"    -> embedded_chunks.json: {len(embedded_chunks):,} -> {len(filtered_embedded):,} chunks")

    print()

    # 5. Clean up PostgreSQL Database
    print("[5] Cleaning up PostgreSQL database...")
    db_conn = get_connection()
    try:
        with db_conn.cursor() as cur:
            # Query kept document IDs
            cur.execute(
                "SELECT id, filename FROM documents WHERE filename = ANY(%s);",
                (list(KEPT_BOOK_FILENAMES),),
            )
            kept_docs = cur.fetchall()
            kept_ids = [r[0] for r in kept_docs]
            print(f"    Kept documents in DB: {kept_docs}")

            # Query non-kept documents
            cur.execute(
                "SELECT id, filename FROM documents WHERE NOT (id = ANY(%s));",
                (kept_ids,),
            )
            deleted_docs = cur.fetchall()
            deleted_ids = [r[0] for r in deleted_docs]
            print(f"    Documents to delete from DB: {deleted_docs}")

            # Count chunks before deletion
            cur.execute("SELECT count(*) FROM chunks WHERE document_id = ANY(%s);", (deleted_ids,))
            del_chunks_count = cur.fetchone()[0]

            # Delete non-kept documents (CASCADE deletes chunks and relationships)
            cur.execute("DELETE FROM documents WHERE id = ANY(%s);", (deleted_ids,))
            print(f"    -> Cascade deleted {len(deleted_ids)} documents and {del_chunks_count:,} chunks from PostgreSQL.")

            # Clean up orphan entities that have no remaining relationships
            cur.execute(
                """
                DELETE FROM entities
                WHERE id NOT IN (
                    SELECT source_entity_id FROM relationships
                    UNION
                    SELECT target_entity_id FROM relationships
                );
                """
            )
            deleted_orphan_entities = cur.rowcount
            print(f"    -> Removed {deleted_orphan_entities:,} orphan graph entities without active relationships.")

        db_conn.commit()
        print("    -> Committed PostgreSQL cleanup transaction successfully.\n")
    finally:
        db_conn.close()

    # 6. Clean up SQLite Checkpoint Database
    print("[6] Cleaning up SQLite checkpoint tracking database...")
    if CHECKPOINT_DB_PATH.exists():
        # Backup checkpoint DB before modification
        shutil.copy2(str(CHECKPOINT_DB_PATH), str(ARCHIVED_DIR / "graph_extraction_progress_backup.db"))
        cp_conn = sqlite3.connect(str(CHECKPOINT_DB_PATH))
        try:
            cur = cp_conn.cursor()
            cur.execute("SELECT count(*) FROM chunk_progress WHERE document_id NOT IN (?, ?);", tuple(kept_ids))
            del_cp_count = cur.fetchone()[0]
            cur.execute("DELETE FROM chunk_progress WHERE document_id NOT IN (?, ?);", tuple(kept_ids))
            cp_conn.commit()
            cur.execute("VACUUM;")
            print(f"    -> Removed {del_cp_count:,} archived chunk progress records from checkpoint DB.")
        finally:
            cp_conn.close()
    print()

    # 7. Verification & Final Reporting
    print("=" * 80)
    print("  POST-CLEANUP VERIFICATION REPORT")
    print("=" * 80)

    # Verify PostgreSQL state
    db_conn = get_connection()
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT id, filename, document_type, department, access_level FROM documents ORDER BY id;")
            active_docs = cur.fetchall()

            cur.execute("SELECT d.filename, count(c.id) FROM documents d JOIN chunks c ON d.id = c.document_id GROUP BY d.filename ORDER BY d.filename;")
            doc_chunk_breakdown = cur.fetchall()

            cur.execute("SELECT count(*) FROM chunks;")
            total_active_chunks = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM entities;")
            total_entities = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM relationships;")
            total_relationships = cur.fetchone()[0]

            cur.execute("SELECT d.filename, count(r.id) FROM documents d LEFT JOIN relationships r ON d.id = r.document_id GROUP BY d.filename ORDER BY d.filename;")
            rel_breakdown = cur.fetchall()
    finally:
        db_conn.close()

    # Verify SQLite checkpoint state
    if CHECKPOINT_DB_PATH.exists():
        cp_conn = sqlite3.connect(str(CHECKPOINT_DB_PATH))
        try:
            cur = cp_conn.cursor()
            cur.execute("SELECT count(*), sum(CASE WHEN status='completed' THEN 1 ELSE 0 END), sum(CASE WHEN status='pending' THEN 1 ELSE 0 END) FROM chunk_progress;")
            cp_tot, cp_comp, cp_pend = cur.fetchone()
        finally:
            cp_conn.close()
    else:
        cp_tot, cp_comp, cp_pend = 0, 0, 0

    print("1. ACTIVE DOCUMENTS IN POSTGRESQL:")
    for doc in active_docs:
        print(f"   - [ID {doc[0]}] {doc[1]} (Type: {doc[2]}, Dept: {doc[3]}, Access: {doc[4]})")

    print("\n2. ACTIVE CHUNK COUNT:")
    print(f"   - Total Active Chunks in PostgreSQL: {total_active_chunks:,}")
    for doc_name, cnt in doc_chunk_breakdown:
        print(f"     * {doc_name}: {cnt:,} chunks")

    print(f"\n3. ARCHIVED BOOKS LOCATION:")
    print(f"   - Path: {ARCHIVED_DIR}")
    print(f"   - Original PDFs in archive:")
    for f in ARCHIVED_DIR.glob("*.pdf"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"     * {f.name} ({size_mb:.2f} MB)")

    print(f"\n4. POSTGRESQL TOTAL COUNTS:")
    print(f"   - Documents Count    : {len(active_docs)}")
    print(f"   - Chunks Count       : {total_active_chunks:,}")

    print(f"\n5. KNOWLEDGE GRAPH COUNTS:")
    print(f"   - Entities Count     : {total_entities:,}")
    print(f"   - Relationships Count: {total_relationships:,}")
    for doc_name, cnt in rel_breakdown:
        print(f"     * {doc_name}: {cnt:,} relationships")

    print(f"\n6. CHECKPOINT PROGRESS STATE:")
    print(f"   - Checkpoint DB Total Chunks: {cp_tot:,}")
    print(f"   - Completed Chunks          : {cp_comp:,}")
    print(f"   - Pending Chunks            : {cp_pend:,}")

    print("\n" + "=" * 80)
    print("  CLEANUP COMPLETE: ONLY THE TWO SELECTED BOOKS REMAIN IN ACTIVE CORPUS")
    print("=" * 80)


if __name__ == "__main__":
    main()
