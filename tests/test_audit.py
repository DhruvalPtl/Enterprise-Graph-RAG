"""
Tests for PDF Extraction Audit Tool (scripts/audit_pdf_extraction.py).
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.audit_pdf_extraction import (
    PageAuditRecord,
    count_page_images,
    audit_single_page,
    inspect_ai_index_page_372,
    save_reports,
)


def test_audit_single_page_clean_text():
    """Verifies that clean plain text without anomalies produces a non-suspicious record."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a clean plain text document explaining retrieval augmented generation."
    mock_page.get.return_value = {}

    record = audit_single_page(mock_page, page_number=1, doc_name="clean.pdf")

    assert record.page_number == 1
    assert record.document == "clean.pdf"
    assert record.char_count > 0
    assert record.word_count > 0
    assert record.nul_count == 0
    assert record.is_sparse is False
    assert record.is_suspicious is False
    assert record.reasons == []


def test_audit_single_page_detects_nul_bytes():
    """Verifies that raw text containing NUL bytes (0x00) is flagged accurately."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Chapter 1: Natural Language Processing\x00\x00 with NUL bytes\x00."
    mock_page.get.return_value = {}

    record = audit_single_page(mock_page, page_number=5, doc_name="corrupt.pdf")

    assert record.nul_count == 3
    assert record.is_suspicious is True
    assert any("NUL" in r for r in record.reasons)


def test_audit_single_page_detects_sparse_page_with_images():
    """Verifies that sparse text paired with image objects is flagged as visual/scan."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Figure 1."
    # Mock resource dictionary containing an image XObject
    mock_page.get.return_value = {
        "/XObject": {
            "/Img1": {"/Subtype": "/Image"}
        }
    }

    record = audit_single_page(mock_page, page_number=12, doc_name="diagram.pdf")

    assert record.is_sparse is True
    assert record.image_count == 1
    assert record.is_suspicious is True
    assert any("Sparse text" in r and "images" in r for r in record.reasons)


def test_count_page_images_from_xobject_dictionary():
    """Verifies image counting from /Resources/XObject dictionary."""
    mock_page = MagicMock()
    mock_page.get.return_value = {
        "/XObject": {
            "/Img1": {"/Subtype": "/Image"},
            "/Img2": {"/Subtype": "/Image"},
            "/Form1": {"/Subtype": "/Form"},
        }
    }
    assert count_page_images(mock_page) == 2


def test_save_reports_creates_valid_json_and_csv(tmp_path: Path):
    """Verifies that save_reports creates well-structured JSON and CSV files."""
    records = [
        PageAuditRecord(
            document="test.pdf",
            page_number=1,
            char_count=500,
            word_count=80,
            image_count=2,
            nul_count=0,
            control_char_count=0,
            replacement_char_count=0,
            is_sparse=False,
            caption_count=1,
            has_clustered_captions=False,
            has_clustered_dates=False,
            reading_order_inversion=False,
            is_suspicious=False,
            reasons=[],
        ),
        PageAuditRecord(
            document="test.pdf",
            page_number=2,
            char_count=30,
            word_count=5,
            image_count=1,
            nul_count=1,
            control_char_count=0,
            replacement_char_count=0,
            is_sparse=True,
            caption_count=0,
            has_clustered_captions=False,
            has_clustered_dates=False,
            reading_order_inversion=True,
            is_suspicious=True,
            reasons=["Contains 1 NUL (0x00) bytes", "Sparse text"],
        ),
    ]

    json_file, csv_file = save_reports(records, output_dir=tmp_path)

    assert json_file.exists()
    assert csv_file.exists()

    import json
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_pages_audited"] == 2
    assert data["suspicious_pages_count"] == 1


def test_deep_inspection_ai_index_page_372():
    """Integration check on raw AI Index page 372 if the file is present."""
    pdf_path = Path("data/raw/Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf")
    if not pdf_path.exists():
        pytest.skip("AI Index PDF not found in data/raw")

    details = inspect_ai_index_page_372(pdf_path)
    assert details["page_number"] == 372
    assert details["image_count"] == 3
    assert details["raw_text_length"] > 1000
    assert len(details["dates_stream_positions"]) >= 2
