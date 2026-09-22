"""
Tests for IngestionPipeline end-to-end processing and storage.
"""
import pytest
import json
from pathlib import Path

from app.pipeline import IngestionPipeline


@pytest.fixture
def sample_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "raw"


def test_pipeline_directory_ingestion(sample_data_dir: Path, tmp_path: Path):
    processed_dir = tmp_path / "processed"
    pipeline = IngestionPipeline(raw_dir=sample_data_dir, processed_dir=processed_dir)

    result = pipeline.process_directory()

    assert result["total_documents"] >= 3  # .pdf, .md, .txt
    assert result["total_chunks"] >= 3
    assert result["average_chunk_size_chars"] > 0
    assert len(result["files_summary"]) == result["total_documents"]

    # Verify all_chunks.json was created
    all_chunks_file = processed_dir / "all_chunks.json"
    assert all_chunks_file.exists()

    with open(all_chunks_file, "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)

    assert len(saved_chunks) == result["total_chunks"]

    # Check structure of the saved elements
    first_chunk = saved_chunks[0]
    assert "chunk_id" in first_chunk
    assert "document_id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    assert "document_name" in first_chunk["metadata"]
    assert "page_number" in first_chunk["metadata"]


def test_pipeline_empty_directory(tmp_path: Path):
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    processed_dir = tmp_path / "processed"

    pipeline = IngestionPipeline(raw_dir=empty_raw, processed_dir=processed_dir)
    result = pipeline.process_directory()

    assert result["total_documents"] == 0
    assert result["total_chunks"] == 0
