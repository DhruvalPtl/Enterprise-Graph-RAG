"""
Graph RAG Phase G2.2: Full-Corpus Extraction and PostgreSQL Persistence Runner.

Processes all 11,609 document chunks through structured Gemini entity/relationship extraction,
checkpoints progress durably in SQLite, and persists graph nodes and edges into PostgreSQL
with ACID transactional safety, full chunk provenance, and rate-limiting backoff.

Usage:
    python scripts/run_graph_extraction_full.py [--workers 2] [--batch-size 25] [--delay 0.3] [--limit 100]
"""
import argparse
import concurrent.futures
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection, insert_chunk_graph_transaction
from app.graph_checkpoint import GraphExtractionCheckpoint, DEFAULT_CHECKPOINT_DB_PATH
from app.graph_extractor import GraphExtractorService, ModelRatePacer
from app.graph_ontology import ValidatedExtractionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("graph_extraction_full")

# Global graceful shutdown flag
_SHUTDOWN_REQUESTED = False


def _sigint_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    print("\n[SHUTDOWN] Interruption signal received. Completing current batch and exiting safely...")
    _SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, _sigint_handler)


def get_all_chunks_from_db() -> List[Tuple[int, int, Optional[int]]]:
    """Retrieves all chunk IDs, document IDs, and page numbers from PostgreSQL ordered by ID."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, document_id, page_number FROM chunks ORDER BY id ASC;")
            return cur.fetchall()
    finally:
        conn.close()


def process_single_chunk(
    chunk_meta: Tuple[int, int, Optional[int]],
    extractor: GraphExtractorService,
    checkpoint: GraphExtractionCheckpoint,
    delay: float = 0.0,
    max_rpm: float = 28.5,
) -> Dict[str, Any]:
    """
    Processes a single chunk: extracts graph data, persists to PostgreSQL in an atomic transaction,
    and updates the SQLite checkpoint. Paces execution to strictly stay within rate limits.
    """
    chunk_id, doc_id, page_number = chunk_meta
    checkpoint.mark_processing(chunk_id)

    start_time = time.time()
    db_conn = get_connection()

    try:
        # Fetch chunk content
        with db_conn.cursor() as cur:
            cur.execute("SELECT content FROM chunks WHERE id = %s;", (chunk_id,))
            row = cur.fetchone()
            if not row or not row[0] or not row[0].strip():
                checkpoint.mark_skipped_empty(chunk_id)
                return {"chunk_id": chunk_id, "status": "skipped_empty", "entities": 0, "relationships": 0}
            content = row[0]

        # Extract using Gemini service (pacing is handled automatically inside extractor)
        res: ValidatedExtractionResult = extractor.extract_from_chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            page_number=page_number,
            chunk_text=content,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        if res.status == "error":
            err_msg = res.validation_errors[0] if res.validation_errors else "Extraction error"
            checkpoint.mark_failed(chunk_id, err_msg)
            return {"chunk_id": chunk_id, "status": "error", "error": err_msg}

        if res.status == "skipped_empty":
            checkpoint.mark_skipped_empty(chunk_id)
            return {"chunk_id": chunk_id, "status": "skipped_empty", "entities": 0, "relationships": 0}

        # Atomically insert into PostgreSQL
        ent_count, rel_count = insert_chunk_graph_transaction(db_conn, res)

        # Mark completed in SQLite checkpoint
        in_tok = getattr(res, "input_tokens", 0) or 0
        out_tok = getattr(res, "output_tokens", 0) or 0
        actual_model = getattr(res, "model_used", None) or extractor.last_used_model or extractor.model_name
        checkpoint.mark_completed(
            chunk_id=chunk_id,
            entity_count=ent_count,
            relationship_count=rel_count,
            model_used=actual_model,
            duration_ms=duration_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

        print(
            f"  [Chunk {chunk_id}] Extracted ({actual_model}): {ent_count} ents, {rel_count} rels | "
            f"Tokens: {in_tok} in / {out_tok} out | {duration_ms}ms",
            flush=True,
        )

        # If extractor has no pacer, enforce manual rate ceiling
        if extractor.pacer is None and max_rpm and max_rpm > 0:
            total_chunk_time = time.time() - start_time
            min_cycle = 60.0 / max_rpm
            sleep_needed = max(0.1, min_cycle - total_chunk_time)
            end_sleep = time.time() + sleep_needed
            while time.time() < end_sleep:
                time.sleep(min(0.2, max(0.02, end_sleep - time.time())))
        elif delay > 0:
            time.sleep(delay)

        return {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "page_number": page_number,
            "status": "completed",
            "entities": ent_count,
            "relationships": rel_count,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "model_used": actual_model,
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        duration_ms = int((time.time() - start_time) * 1000)
        err_msg = str(exc)
        logger.error(f"Error processing chunk {chunk_id}: {err_msg}")
        checkpoint.mark_failed(chunk_id, err_msg)
        return {"chunk_id": chunk_id, "status": "failed", "error": err_msg}

    finally:
        db_conn.close()


def generate_full_reports(
    checkpoint: GraphExtractionCheckpoint,
    total_chunks: int,
    elapsed_time: float,
    reports_dir: Path,
):
    """Generates comprehensive JSON and Markdown extraction reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = checkpoint.get_progress_summary()
    failed_chunks = checkpoint.get_failed_chunks()

    # Query PostgreSQL for unique entities and relationships count
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM entities;")
            unique_entities = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM relationships;")
            total_relationships = cur.fetchone()[0]

            cur.execute(
                """
                SELECT entity_type, count(*) as cnt
                FROM entities
                GROUP BY entity_type
                ORDER BY cnt DESC;
                """
            )
            entity_type_dist = {r[0]: r[1] for r in cur.fetchall()}

            cur.execute(
                """
                SELECT relationship_type, count(*) as cnt
                FROM relationships
                GROUP BY relationship_type
                ORDER BY cnt DESC;
                """
            )
            rel_type_dist = {r[0]: r[1] for r in cur.fetchall()}

            # Top degree entities
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
                LIMIT 20;
                """
            )
            top_degree = [
                {"name": r[0], "type": r[1], "degree": r[2]}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()

    # Query model distribution from checkpoint DB
    model_dist = {}
    try:
        with checkpoint._connection() as s_conn:
            s_cur = s_conn.cursor()
            s_cur.execute("SELECT COALESCE(model_used, 'unknown'), count(*) FROM chunk_progress WHERE status = 'completed' GROUP BY model_used;")
            model_dist = {r[0]: r[1] for r in s_cur.fetchall()}
    except Exception as exc:
        logger.warning(f"Could not query model distribution: {exc}")

    json_payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_corpus_chunks": total_chunks,
            "completed_chunks": summary["completed"],
            "failed_chunks": summary["failed"],
            "skipped_empty_chunks": summary["skipped_empty"],
            "pending_chunks": summary["pending"],
            "total_attempts": summary["total_attempts"],
            "total_input_tokens": summary.get("total_input_tokens", 0),
            "total_output_tokens": summary.get("total_output_tokens", 0),
            "total_tokens": summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0),
            "elapsed_seconds": round(elapsed_time, 2),
            "average_seconds_per_chunk": round(elapsed_time / max(1, summary["completed"]), 3),
            "unique_entities_in_db": unique_entities,
            "total_relationships_in_db": total_relationships,
            "model_distribution": model_dist,
            "entity_type_distribution": entity_type_dist,
            "relationship_type_distribution": rel_type_dist,
            "top_connected_entities": top_degree,
        }
    }

    # Save JSON report
    json_path = reports_dir / "graph_extraction_full_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[REPORT] Saved full extraction JSON -> {json_path}")

    # Save failed chunks JSON if any
    if failed_chunks:
        failed_path = reports_dir / "graph_extraction_failed_chunks.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_chunks, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved {len(failed_chunks)} permanently failed chunks -> {failed_path}")

    # Generate Markdown Report
    total_tok = summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0)
    md_lines = [
        "# Graph RAG Phase G2.2: Full-Corpus Extraction & Persistence Report",
        "",
        f"**Date**: {datetime.now().isoformat()}",
        f"**Corpus Size**: {total_chunks:,} chunks across active documents",
        f"**Database**: PostgreSQL `rag_db` (`entities`, `relationships`)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Production Status",
        "",
        "| Metric | Value | Production Status |",
        "| :--- | :--- | :--- |",
        f"| **Total Chunks in Corpus** | **{total_chunks:,}** | 100% Accounted |",
        f"| **Completed Chunks** | **{summary['completed']:,}** | Extracted & Persisted |",
        f"| **Skipped / Empty Chunks** | **{summary['skipped_empty']:,}** | Clean Whitespace/Sparse |",
        f"| **Permanently Failed Chunks** | **{summary['failed']:,}** | Logged to JSON |",
        f"| **Total Extracted Entities (Unique)** | **{unique_entities:,}** | Deduplicated via Canonicalization |",
        f"| **Total Extracted Relationships** | **{total_relationships:,}** | Persisted with Full Provenance |",
        f"| **Total Input Tokens** | **{summary.get('total_input_tokens', 0):,}** | Gemini Prompt Tokens |",
        f"| **Total Output Tokens** | **{summary.get('total_output_tokens', 0):,}** | Gemini Candidate Tokens |",
        f"| **Total LLM Tokens** | **{total_tok:,}** | Combined Consumption |",
        f"| **Total Execution Duration** | **{str(timedelta(seconds=int(elapsed_time)))}** | Wall clock time |",
        f"| **Average Throughput** | **{summary['completed'] / max(1, elapsed_time):.2f} chunks/sec** | Multi-Account Round-Robin |",
        "",
        "---",
        "",
        "## 2. Entity Distribution by Controlled Type (15 Ontology Types)",
        "",
        "| Entity Type | Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]

    for et, cnt in entity_type_dist.items():
        pct = (cnt / max(1, unique_entities)) * 100
        md_lines.append(f"| `{et}` | {cnt:,} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Relationship Distribution by Controlled Type (19 Ontology Types)",
        "",
        "| Relationship Type | Count | Percentage |",
        "| :--- | :--- | :--- |",
    ])

    for rt, cnt in rel_type_dist.items():
        pct = (cnt / max(1, total_relationships)) * 100
        md_lines.append(f"| `{rt}` | {cnt:,} | {pct:.1f}% |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Top 20 Most Connected Hub Entities (Degree Centrality)",
        "",
        "| Entity Name | Entity Type | Degree (In + Out Edges) |",
        "| :--- | :--- | :--- |",
    ])

    for hub in top_degree:
        md_lines.append(f"| **{hub['name']}** | `{hub['type']}` | {hub['degree']:,} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Provenance & Access Control Neutrality",
        "",
        "- Every relationship row in `relationships` contains valid non-null foreign keys `document_id` and `chunk_id`.",
        "- Downstream Graph RAG queries join against `documents` to enforce `department` and `access_level` filters directly on edge traversal.",
        "- Zero modifications were made to the existing vector/BM25/Reranker/FastAPI retrieval pipeline.",
        "",
    ])

    md_path = reports_dir / "graph_extraction_full_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[REPORT] Saved full extraction Markdown -> {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Graph RAG Phase G2.2 Full Corpus Extractor")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent worker threads (default: 1)")
    parser.add_argument("--max-rpm", type=float, default=28.5, help="Combined target requests per minute across all models (default: 28.5)")
    parser.add_argument("--max-model-rpm", type=float, default=14.5, help="Target max requests per minute per individual model (default: 14.5)")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional extra delay in seconds (default: 0.0)")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for checkpoint queries (default: 25)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts per chunk (default: 3)")
    parser.add_argument("--models", type=str, default="gemini-3.5-flash-lite,gemini-3.1-flash-lite", help="Comma-separated LLM models to alternate (default: gemini-3.5-flash-lite,gemini-3.1-flash-lite)")
    parser.add_argument("--model", type=str, default=None, help="Single extraction model override (if set, overrides --models)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max chunks to process in this run")
    parser.add_argument("--retry-failed", action="store_true", help="Reset previously failed chunks back to pending")
    parser.add_argument("--checkpoint-db", type=str, default=str(DEFAULT_CHECKPOINT_DB_PATH), help="Path to checkpoint SQLite DB")
    args = parser.parse_args()

    # Parse models list
    if args.model:
        models_list = [args.model.strip()]
    else:
        models_list = [m.strip() for m in args.models.split(",") if m.strip()]

    # Initialize ModelRatePacer
    pacer = ModelRatePacer(max_rpm_per_model=args.max_model_rpm, combined_max_rpm=args.max_rpm)

    # Initialize Extractor Service with dual models and rate pacer
    extractor = GraphExtractorService(models=models_list, pacer=pacer)

    print("=" * 80)
    print("  Graph RAG Phase G2.2: Full-Corpus Extraction (Dual-Model Interleaved Mode)")
    print("=" * 80)
    print(f"  Worker Threads       : {args.workers} (1 active account at a time)")
    print(f"  Interleaved Models   : {extractor.models}")
    print(f"  Max Combined Rate    : Up to {args.max_rpm:.1f} RPM (Min interval: {60.0/args.max_rpm:.2f}s)")
    print(f"  Max Per-Model Rate   : Up to {args.max_model_rpm:.1f} RPM (Min per-model: {60.0/args.max_model_rpm:.2f}s)")
    print(f"  Account Quota        : 500 RPD on 3.5 + 500 RPD on 3.1 = 1,000 RPD per account")
    print(f"  Batch Size           : {args.batch_size}")
    print(f"  Max Retries / Chunk  : {args.max_retries}")
    print(f"  Checkpoint Database  : {args.checkpoint_db}")
    if args.limit:
        print(f"  Limit for this run   : {args.limit} chunks")
    print("-" * 80)

    # 1. Initialize Checkpoint Tracker
    checkpoint = GraphExtractionCheckpoint(Path(args.checkpoint_db))

    # 2. Query all chunks from PostgreSQL
    print("Fetching corpus chunk metadata from PostgreSQL...")
    all_chunks = get_all_chunks_from_db()
    total_corpus_count = len(all_chunks)
    print(f"Total chunks in corpus: {total_corpus_count:,}")

    # 3. Register all chunks into SQLite checkpoint
    new_reg = checkpoint.register_chunks(all_chunks)
    if new_reg > 0:
        print(f"Registered {new_reg:,} new chunks into checkpoint tracking table.")
    else:
        print("Existing checkpoint found. Resuming extraction.")

    # 4. Recover any dangling processing chunks
    reset_cnt = checkpoint.reset_processing_to_pending()
    if reset_cnt > 0:
        print(f"Reset {reset_cnt} interrupted chunks from 'processing' to 'pending'.")

    # 4b. Optionally reset failed chunks
    if args.retry_failed:
        re_cnt = checkpoint.reset_failed_to_pending()
        if re_cnt > 0:
            print(f"Reset {re_cnt} previously failed chunks back to 'pending'.")

    # Initial summary
    initial_summary = checkpoint.get_progress_summary()
    print(f"Initial State: Completed={initial_summary['completed']:,}, Pending={initial_summary['pending']:,}, Failed={initial_summary['failed']:,}\n")

    # 5. Key Pool Configuration Status
    if extractor.key_pool:
        active = extractor.key_pool.active_account
        print(f"Loaded {extractor.key_pool.total_accounts} accounts (Sequential Single-Account Mode):")
        for acc in extractor.key_pool.accounts:
            stat = " [DISABLED]" if acc.is_disabled else (" [ACTIVE]" if acc == active else "")
            print(f"  [{acc.account_id}] {acc.name:<16} key={acc.masked_key}{stat}")
        print(f"\n  Active Starting Account: {active.name if active else 'None'}")
        print("  Policy: Requests exclusively use the active account, alternating between dual models.")
        print("          Upon rate limit / quota exhaustion of both models, it shifts to the next available account.")
    print("-" * 80 + "\n")

    total_processed_in_run = 0
    start_wall_time = time.time()

    try:
        while not _SHUTDOWN_REQUESTED:
            # Check if limit reached
            if args.limit and total_processed_in_run >= args.limit:
                print(f"\n[LIMIT REACHED] Processed {total_processed_in_run} chunks matching --limit {args.limit}.")
                break

            # Fetch next batch
            remaining_limit = (args.limit - total_processed_in_run) if args.limit else args.batch_size
            fetch_size = min(args.batch_size, remaining_limit)
            batch = checkpoint.get_pending_chunks(batch_size=fetch_size, max_attempts=args.max_retries)

            if not batch:
                print("\n[COMPLETE] No pending chunks remaining in checkpoint database!")
                break

            # Process batch concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_single_chunk, chunk_meta, extractor, checkpoint, args.delay, args.max_rpm): chunk_meta[0]
                    for chunk_meta in batch
                }

                for future in concurrent.futures.as_completed(futures):
                    if _SHUTDOWN_REQUESTED:
                        break
                    res = future.result()
                    total_processed_in_run += 1

            # Observability & Progress Display
            curr_summary = checkpoint.get_progress_summary()
            elapsed = time.time() - start_wall_time
            rate = total_processed_in_run / max(0.1, elapsed)
            rem_chunks = curr_summary["pending"]
            eta_seconds = int(rem_chunks / rate) if rate > 0 else 0
            eta_str = str(timedelta(seconds=eta_seconds))

            pct = (curr_summary["completed"] / max(1, total_corpus_count)) * 100
            print(
                f"[{curr_summary['completed']:,}/{total_corpus_count:,}] ({pct:.1f}%) | "
                f"Done: {curr_summary['completed']:,} | "
                f"Fail: {curr_summary['failed']} | "
                f"Ent: {curr_summary['total_entities']:,} | "
                f"Rel: {curr_summary['total_relationships']:,} | "
                f"Rate: {rate:.2f} ch/s | "
                f"ETA: {eta_str}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Extraction paused by user. Progress is durably saved in checkpoint.")

    finally:
        total_elapsed = time.time() - start_wall_time
        reports_dir = BASE_DIR / "reports"
        generate_full_reports(checkpoint, total_corpus_count, total_elapsed, reports_dir)

        # Print final token and key pool summary
        final_sum = checkpoint.get_progress_summary()
        in_t = final_sum.get("total_input_tokens", 0)
        out_t = final_sum.get("total_output_tokens", 0)
        print("\n" + "=" * 80)
        print("  EXTRACTION SUMMARY & TOKEN CONSUMPTION")
        print("=" * 80)
        print(f"  Total Chunks Completed: {final_sum['completed']:,} / {total_corpus_count:,}")
        print(f"  Total Input Tokens    : {in_t:,}")
        print(f"  Total Output Tokens   : {out_t:,}")
        print(f"  Total Tokens (Prompt+Candidates): {in_t + out_t:,}")
        if extractor.key_pool:
            print("\n  Key Pool Account Utilization:")
            for stat in extractor.key_pool.get_pool_status():
                print(f"    - {stat['name']:<16} ({stat['masked_key']}): {stat['requests_served']} requests served, {stat['rate_limits_hit']} rate limits hit")
        print("=" * 80)


if __name__ == "__main__":
    main()
