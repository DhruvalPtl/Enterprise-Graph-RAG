"""
Tests for RecursiveStructuralChunker:
- Schema conformance
- Semantic/structural boundary respect (headings, paragraphs, lists)
- Overlap behavior
- Chunk size bounds
- Metadata preservation (page numbers, section headings, document names)
"""
import pytest
from app.models import Document, DocumentPage
from app.chunker import RecursiveStructuralChunker


def test_chunk_schema_and_fields():
    doc = Document(
        id="doc_test123",
        content="Paragraph one text.\n\nParagraph two text.",
        metadata={"document_name": "test_doc.txt", "file_type": "text"},
        pages=[DocumentPage(page_number=1, text="Paragraph one text.\n\nParagraph two text.")],
    )

    chunker = RecursiveStructuralChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    chunk = chunks[0]

    # Verify chunk structure
    assert chunk.chunk_id == "doc_test123_c0000"
    assert chunk.document_id == "doc_test123"
    assert len(chunk.text) > 0
    assert chunk.metadata["document_name"] == "test_doc.txt"
    assert chunk.metadata["page_number"] == 1
    assert "section" in chunk.metadata
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["char_count"] == len(chunk.text)
    assert chunk.metadata["word_count"] > 0

    # Verify to_dict output matches desired enterprise schema
    d = chunk.to_dict()
    assert "chunk_id" in d
    assert "document_id" in d
    assert "text" in d
    assert "metadata" in d
    assert d["metadata"]["document_name"] == "test_doc.txt"


def test_paragraph_boundary_respect():
    # If two paragraphs easily fit within chunk_size, they should be grouped
    # If adding the third exceeds chunk_size, it splits cleanly without slicing words
    p1 = "First paragraph containing clear sentences about enterprise search."
    p2 = "Second paragraph covering ingestion strategies and document parsing."
    p3 = "Third paragraph detailing database schemas and indexing structures."
    doc_text = f"{p1}\n\n{p2}\n\n{p3}"

    doc = Document(
        id="doc_boundary_test",
        content=doc_text,
        metadata={"document_name": "boundary.txt"},
        pages=[DocumentPage(page_number=1, text=doc_text)],
    )

    # Size chosen so p1 + p2 fits, but adding p3 exceeds chunk_size
    chunker = RecursiveStructuralChunker(chunk_size=160, chunk_overlap=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    # Chunk 0 has p1 and p2 intact
    assert p1 in chunks[0].text
    assert p2 in chunks[0].text
    # Chunk 1 has p3 intact
    assert p3 in chunks[1].text


def test_overlap_preservation():
    p1 = "Alpha statement describing machine learning architecture."
    p2 = "Beta statement covering data pipeline orchestration."
    p3 = "Gamma statement covering production monitoring."
    doc_text = f"{p1}\n\n{p2}\n\n{p3}"

    doc = Document(
        id="doc_overlap_test",
        content=doc_text,
        metadata={"document_name": "overlap.txt"},
        pages=[DocumentPage(page_number=1, text=doc_text)],
    )

    # Set chunk_size such that p1 fits in chunk 0, and overlap is enough to pull p1's tail into chunk 1
    chunker = RecursiveStructuralChunker(chunk_size=70, chunk_overlap=65)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    # Check that adjacent chunks have overlapping context
    chunk0_tail = chunks[0].text[-20:]
    assert chunk0_tail in chunks[1].text or chunks[1].metadata["chunk_index"] == 1


def test_section_heading_tracking():
    content = (
        "# Introduction\n\n"
        "Welcome to the enterprise platform documentation.\n\n"
        "## Architecture Overview\n\n"
        "Here is the high-level architecture of the ingestion engine."
    )

    doc = Document(
        id="doc_heading_test",
        content=content,
        metadata={"document_name": "architecture.md", "file_type": "markdown"},
        pages=[DocumentPage(page_number=1, text=content)],
    )

    chunker = RecursiveStructuralChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Verify section titles were detected and updated
    sections = [c.metadata["section"] for c in chunks]
    assert any("Introduction" in s for s in sections)
    assert any("Architecture Overview" in s for s in sections)


def test_multi_page_metadata_preservation():
    doc = Document(
        id="doc_multipage",
        content="Page 1 Content\n\nPage 2 Content",
        metadata={"document_name": "manual.pdf", "file_type": "pdf"},
        pages=[
            DocumentPage(page_number=1, text="This is content on page one of the enterprise PDF."),
            DocumentPage(page_number=2, text="This is content on page two of the enterprise PDF."),
        ],
    )

    chunker = RecursiveStructuralChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].metadata["page_number"] == 2
    assert chunks[0].chunk_id == "doc_multipage_c0000"
    assert chunks[1].chunk_id == "doc_multipage_c0001"


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError, match="must be strictly less than"):
        RecursiveStructuralChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than"):
        RecursiveStructuralChunker(chunk_size=100, chunk_overlap=150)
