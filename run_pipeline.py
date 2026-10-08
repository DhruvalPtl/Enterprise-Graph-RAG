"""
Command-line entry point to execute the Document Ingestion & Chunking Pipeline.

Usage:
    python run_pipeline.py
"""
import sys
from pathlib import Path
from app.pipeline import IngestionPipeline
from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def main():
    print("=" * 70)
    print("  Enterprise RAG - Document Ingestion & Chunking")
    print("=" * 70)
    print(f"  Source Directory : {RAW_DATA_DIR}")
    print(f"  Target Directory : {PROCESSED_DATA_DIR}")
    print(f"  Target Chunk Size: {DEFAULT_CHUNK_SIZE} chars")
    print(f"  Chunk Overlap    : {DEFAULT_CHUNK_OVERLAP} chars")
    print("-" * 70)

    pipeline = IngestionPipeline()
    result = pipeline.process_directory()

    print("\n[INGESTION SUMMARY]")
    print(f"  Total Documents Processed : {result['total_documents']}")
    print(f"  Total Chunks Generated    : {result['total_chunks']}")
    print(f"  Average Chunk Size        : {result['average_chunk_size_chars']} chars")

    if result.get("files_summary"):
        print("\n[PROCESSED FILES BREAKDOWN]")
        header = f"  {'File Name':<35} | {'Type':<6} | {'Chunks':<6} | {'Characters':<10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for item in result["files_summary"]:
            print(f"  {item['file_name']:<35} | {item['file_type']:<6} | {item['chunks_count']:<6} | {item['total_chars']:<10}")

    print("-" * 70)
    print(f"Output files stored in: {result['processed_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
