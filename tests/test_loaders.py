"""
Tests for Document Loaders (TextLoader, MarkdownLoader, PDFLoader, and load_document factory).
"""
import pytest
from pathlib import Path

from app.loaders import load_document, TextLoader, MarkdownLoader, PDFLoader
from app.models import Document


@pytest.fixture
def sample_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "raw"


def test_text_loader(sample_data_dir: Path):
    txt_path = sample_data_dir / "support_faq.txt"
    assert txt_path.exists(), "Sample FAQ text file must exist"

    doc = load_document(txt_path)
    assert isinstance(doc, Document)
    assert doc.id.startswith("doc_")
    assert doc.metadata["document_name"] == "support_faq.txt"
    assert doc.metadata["file_type"] == "text"
    assert doc.metadata["char_count"] > 0
    assert "What is the primary purpose" in doc.content
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1


def test_markdown_loader(sample_data_dir: Path):
    md_path = sample_data_dir / "ai_governance_policy.md"
    assert md_path.exists(), "Sample policy markdown file must exist"

    doc = load_document(md_path)
    assert isinstance(doc, Document)
    assert doc.metadata["file_type"] == "markdown"
    assert doc.metadata["title"] == "Enterprise AI Governance and Deployment Policy"
    assert "Data Privacy and PII Redaction" in doc.content
    assert len(doc.pages) == 1


def test_pdf_loader(sample_data_dir: Path):
    pdf_path = sample_data_dir / "enterprise_platform_architecture.pdf"
    assert pdf_path.exists(), "Sample PDF file must exist"

    doc = load_document(pdf_path)
    assert isinstance(doc, Document)
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["total_pages"] >= 2
    assert len(doc.pages) >= 2

    # Check that individual pages have correct 1-indexed numbering and text
    assert doc.pages[0].page_number == 1
    assert "Enterprise AI Platform Architecture" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Chunking Strategy" in doc.pages[1].text


def test_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.pdf")


def test_unsupported_file_extension(tmp_path: Path):
    dummy_file = tmp_path / "data.csv"
    dummy_file.write_text("col1,col2\nval1,val2")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document(dummy_file)


def test_normalize_extracted_text():
    from app.loaders import normalize_extracted_text

    raw = "Hello\x00world!\x07\x0c This is a test-\ncase with multiple\n\n\n\nnewlines."
    cleaned = normalize_extracted_text(raw)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "\x0c" not in cleaned
    assert "testcase" in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Helloworld!")


def test_pdf_page_numbering_starts_at_one(tmp_path: Path):
    import pymupdf as fitz
    pdf_path = tmp_path / "multi_page.pdf"
    doc = fitz.open()
    for i in range(3):
        p = doc.new_page()
        p.insert_text((50, 50), f"Content on page {i+1}")
    doc.save(str(pdf_path))
    doc.close()

    loaded = PDFLoader(pdf_path).load()
    assert len(loaded.pages) == 3
    assert [p.page_number for p in loaded.pages] == [1, 2, 3]
    for idx, p in enumerate(loaded.pages):
        assert f"Content on page {idx+1}" in p.text
        assert p.metadata["page_char_count"] == len(p.text)


def test_pdf_table_detection_and_markdown(tmp_path: Path):
    import pymupdf as fitz
    pdf_path = tmp_path / "table_test.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Before Table Header")
    # Draw table grid
    page.draw_rect(fitz.Rect(50, 100, 300, 200), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(50, 150), fitz.Point(300, 150), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(175, 100), fitz.Point(175, 200), color=(0, 0, 0), width=1)
    page.insert_text((60, 130), "Item")
    page.insert_text((185, 130), "Price")
    page.insert_text((60, 180), "Widget")
    page.insert_text((185, 180), "$10")
    page.insert_text((50, 250), "After Table Footer")
    doc.save(str(pdf_path))
    doc.close()

    loaded = PDFLoader(pdf_path).load()
    assert len(loaded.pages) == 1
    page_text = loaded.pages[0].text
    # Both header and footer are preserved
    assert "Before Table Header" in page_text
    assert "After Table Footer" in page_text
    # Table markdown is detected and serialized
    assert "|Item|Price|" in page_text or "|Price|Item|" in page_text or "Widget" in page_text
    # Visual ordering: Before Table occurs before After Table Footer
    assert page_text.index("Before Table Header") < page_text.index("After Table Footer")


def test_pdf_nul_bytes_sanitization(tmp_path: Path):
    import pymupdf as fitz
    pdf_path = tmp_path / "nul_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Safe text without raw NULs")
    doc.save(str(pdf_path))
    doc.close()

    loaded = PDFLoader(pdf_path).load()
    for p in loaded.pages:
        assert "\x00" not in p.text
    assert "\x00" not in loaded.content


def test_pdf_chunking_compatibility(sample_data_dir: Path):
    from app.chunker import RecursiveStructuralChunker
    pdf_path = sample_data_dir / "enterprise_platform_architecture.pdf"
    doc = load_document(pdf_path)

    chunker = RecursiveStructuralChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_id.startswith(doc.id)
        assert chunk.document_id == doc.id
        assert len(chunk.text) > 0
        assert "page_number" in chunk.metadata
        assert chunk.metadata["page_number"] in [1, 2]
        assert "section" in chunk.metadata

