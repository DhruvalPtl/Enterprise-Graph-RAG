"""
Run Knowledge Graph Extraction for a Specified Number of Chunks with Token Usage Tracking.

Processes chunks sequentially or with controlled concurrency, tracks input (prompt) and output
(completion) tokens per request directly from the Gemini API response usage metadata, persists
the graph into PostgreSQL, updates the SQLite checkpoint, and outputs request-level and aggregate
token usage metrics.

Usage:
    python scripts/run_chunks_with_token_tracking.py [--count 20] [--workers 1] [--delay 0.75]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.db import get_connection, insert_chunk_graph_transaction
from app.graph_checkpoint import GraphExtractionCheckpoint, DEFAULT_CHECKPOINT_DB_PATH
from app.graph_extractor import GraphExtractorService
from app.graph_ontology import ValidatedExtractionResult


def run_token_tracking_extraction(
    count: int = 20,
    workers: int = 1,
    delay: float = 0.75,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"  Graph Extraction with Token Usage Tracking: {count} Chunks")
    print("=" * 80)
    print(f"  Target Chunks Count  : {count}")
    print(f"  Concurrent Workers   : {workers}")
    print(f"  Inter-Request Delay  : {delay}s")
    print(f"  Checkpoint Database  : {DEFAULT_CHECKPOINT_DB_PATH}")
    print("-" * 80)

    checkpoint = GraphExtractionCheckpoint(DEFAULT_CHECKPOINT_DB_PATH)
    checkpoint.reset_processing_to_pending()

    # Retrieve next batch of pending chunks
    pending_batch = checkpoint.get_pending_chunks(batch_size=count, max_attempts=3)
    if not pending_batch:
        print("[ERROR] No pending chunks found in checkpoint database!")
        return {}

    actual_count = len(pending_batch)
    print(f"Retrieved {actual_count} pending chunks from checkpoint database.\n")

    extractor = GraphExtractorService()
    print(f"Initialized GraphExtractorService (Model: {extractor.model_name})\n")

    request_records: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_entities_extracted = 0
    total_relationships_extracted = 0

    start_wall_time = time.time()

    for idx, chunk_meta in enumerate(pending_batch, start=1):
        chunk_id, doc_id, page_number = chunk_meta
        checkpoint.mark_processing(chunk_id)

        req_start = time.time()
        db_conn = get_connection()

        try:
            # 1. Fetch chunk content and document filename
            with db_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.content, d.filename 
                    FROM chunks c 
                    JOIN documents d ON c.document_id = d.id 
                    WHERE c.id = %s;
                    """,
                    (chunk_id,),
                )
                row = cur.fetchone()
                if not row or not row[0] or not row[0].strip():
                    checkpoint.mark_skipped_empty(chunk_id)
                    rec = {
                        "request_number": idx,
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "page_number": page_number,
                        "filename": row[1] if row else "unknown",
                        "status": "skipped_empty",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "entities_count": 0,
                        "relationships_count": 0,
                        "latency_seconds": round(time.time() - req_start, 3),
                        "model_used": extractor.model_name,
                    }
                    request_records.append(rec)
                    print(f"[{idx:02d}/{actual_count:02d}] Chunk {chunk_id} (Doc {doc_id}, P.{page_number}): SKIPPED (Empty content)")
                    continue

                content = row[0]
                filename = row[1]

            # 2. Perform Extraction via Gemini
            res: ValidatedExtractionResult = extractor.extract_from_chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                page_number=page_number,
                chunk_text=content,
            )

            req_latency = round(time.time() - req_start, 3)

            if res.status == "error":
                err_msg = res.validation_errors[0] if res.validation_errors else "Extraction error"
                checkpoint.mark_failed(chunk_id, err_msg)
                rec = {
                    "request_number": idx,
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "page_number": page_number,
                    "filename": filename,
                    "status": "error",
                    "error_message": err_msg,
                    "input_tokens": getattr(res, "input_tokens", 0) or 0,
                    "output_tokens": getattr(res, "output_tokens", 0) or 0,
                    "total_tokens": (getattr(res, "input_tokens", 0) or 0) + (getattr(res, "output_tokens", 0) or 0),
                    "entities_count": 0,
                    "relationships_count": 0,
                    "latency_seconds": req_latency,
                    "model_used": extractor.model_name,
                }
                request_records.append(rec)
                print(f"[{idx:02d}/{actual_count:02d}] Chunk {chunk_id} (Doc {doc_id}, P.{page_number}): ERROR ({err_msg})")
                continue

            # 3. Atomically persist into PostgreSQL
            ent_count, rel_count = insert_chunk_graph_transaction(db_conn, res)

            # 4. Mark completed in SQLite checkpoint with token usage
            in_tok = getattr(res, "input_tokens", 0) or 0
            out_tok = getattr(res, "output_tokens", 0) or 0
            tot_tok = in_tok + out_tok

            checkpoint.mark_completed(
                chunk_id=chunk_id,
                entity_count=ent_count,
                relationship_count=rel_count,
                model_used=extractor.model_name,
                duration_ms=int(req_latency * 1000),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_entities_extracted += ent_count
            total_relationships_extracted += rel_count

            rec = {
                "request_number": idx,
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "page_number": page_number,
                "filename": filename,
                "status": "completed",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "entities_count": ent_count,
                "relationships_count": rel_count,
                "latency_seconds": req_latency,
                "model_used": extractor.model_name,
            }
            request_records.append(rec)

            print(
                f"[{idx:02d}/{actual_count:02d}] Chunk {chunk_id:<5} | "
                f"Page {page_number:<3} | "
                f"Input: {in_tok:>5,} tok | "
                f"Output: {out_tok:>4,} tok | "
                f"Total: {tot_tok:>5,} tok | "
                f"Ent: {ent_count:>2} | Rel: {rel_count:>2} | "
                f"{req_latency:>5.2f}s",
                flush=True,
            )

            if delay > 0 and idx < actual_count:
                time.sleep(delay)

        except Exception as exc:
            req_latency = round(time.time() - req_start, 3)
            err_msg = str(exc)
            checkpoint.mark_failed(chunk_id, err_msg)
            rec = {
                "request_number": idx,
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "page_number": page_number,
                "status": "failed",
                "error_message": err_msg,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "entities_count": 0,
                "relationships_count": 0,
                "latency_seconds": req_latency,
                "model_used": extractor.model_name,
            }
            request_records.append(rec)
            print(f"[{idx:02d}/{actual_count:02d}] Chunk {chunk_id}: FAILED ({err_msg})")

        finally:
            db_conn.close()

    total_wall_time = round(time.time() - start_wall_time, 2)
    successful_requests = [r for r in request_records if r["status"] == "completed"]
    success_count = len(successful_requests)
    combined_tokens = total_input_tokens + total_output_tokens

    avg_input = round(total_input_tokens / max(1, success_count), 1)
    avg_output = round(total_output_tokens / max(1, success_count), 1)
    avg_total = round(combined_tokens / max(1, success_count), 1)
    avg_latency = round(sum(r["latency_seconds"] for r in successful_requests) / max(1, success_count), 2)

    results_summary = {
        "metadata": {
            "requested_chunks": count,
            "processed_chunks": actual_count,
            "successful_chunks": success_count,
            "failed_chunks": actual_count - success_count,
            "model_used": extractor.model_name,
            "total_wall_time_seconds": total_wall_time,
            "throughput_chunks_per_second": round(actual_count / max(0.1, total_wall_time), 3),
        },
        "token_totals": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_combined_tokens": combined_tokens,
        },
        "token_averages_per_chunk": {
            "average_input_tokens": avg_input,
            "average_output_tokens": avg_output,
            "average_combined_tokens": avg_total,
            "average_latency_seconds": avg_latency,
        },
        "graph_totals": {
            "total_entities_extracted": total_entities_extracted,
            "total_relationships_extracted": total_relationships_extracted,
        },
        "request_details": request_records,
    }

    # Print Formatted Report Table
    print("\n" + "=" * 105)
    print("  DETAILED REQUEST-LEVEL TOKEN USAGE TABLE (20 CHUNKS)")
    print("=" * 105)
    print(
        f"{'Req':<4} | {'Chunk ID':<8} | {'Doc':<4} | {'Page':<5} | {'Status':<10} | "
        f"{'Input Tokens':<12} | {'Output Tokens':<13} | {'Total Tokens':<12} | {'Ent':<4} | {'Rel':<4} | {'Latency':<7}"
    )
    print("-" * 105)
    for r in request_records:
        print(
            f"{r['request_number']:<4} | "
            f"{r['chunk_id']:<8} | "
            f"{r['document_id']:<4} | "
            f"{str(r['page_number']):<5} | "
            f"{r['status']:<10} | "
            f"{r['input_tokens']:>12,} | "
            f"{r['output_tokens']:>13,} | "
            f"{r['total_tokens']:>12,} | "
            f"{r['entities_count']:>4} | "
            f"{r['relationships_count']:>4} | "
            f"{r['latency_seconds']:>6.2f}s"
        )
    print("-" * 105)
    print(
        f"{'TOTAL':<36} | "
        f"{total_input_tokens:>12,} | "
        f"{total_output_tokens:>13,} | "
        f"{combined_tokens:>12,} | "
        f"{total_entities_extracted:>4} | "
        f"{total_relationships_extracted:>4} | "
        f"{total_wall_time:>6.2f}s"
    )
    print("=" * 105)

    print("\n" + "=" * 60)
    print("  AGGREGATE TOKEN USAGE SUMMARY")
    print("=" * 60)
    print(f"  Total Chunks Processed       : {actual_count}")
    print(f"  Successful Extractions       : {success_count} / {actual_count}")
    print(f"  Total Input (Prompt) Tokens  : {total_input_tokens:,}")
    print(f"  Total Output (Completion)    : {total_output_tokens:,}")
    print(f"  Total Combined Tokens        : {combined_tokens:,}")
    print("-" * 60)
    print(f"  Average Input Tokens / Chunk : {avg_input:,}")
    print(f"  Average Output Tokens / Chunk: {avg_output:,}")
    print(f"  Average Total Tokens / Chunk : {avg_total:,}")
    print(f"  Average Request Latency      : {avg_latency:.2f}s")
    print(f"  Total Execution Time         : {total_wall_time:.2f}s")
    print(f"  Total Entities Extracted     : {total_entities_extracted:,}")
    print(f"  Total Relationships Extracted: {total_relationships_extracted:,}")
    print("=" * 60 + "\n")

    # Save to JSON
    out_file = output_json or (BASE_DIR / "reports" / f"token_tracking_{count}_chunks.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved token tracking JSON report to: {out_file}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Run Extraction for Chunks with Token Usage Tracking")
    parser.add_argument("--count", type=int, default=20, help="Number of chunks to process (default: 20)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default: 1)")
    parser.add_argument("--delay", type=float, default=0.75, help="Inter-request delay in seconds (default: 0.75)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    run_token_tracking_extraction(
        count=args.count,
        workers=args.workers,
        delay=args.delay,
        output_json=out_path,
    )


if __name__ == "__main__":
    main()
