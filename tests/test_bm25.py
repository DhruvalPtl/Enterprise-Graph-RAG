"""
Unit and Integration Tests for BM25 Lexical Retrieval.
"""
import pytest
from app.bm25 import BM25Retriever, tokenize, search_bm25


def test_tokenize_basic():
    """Verifies lowercasing, punctuation stripping, and token extraction."""
    text = "Enterprise AI: RAG pipelines, chunk-size, and multi-page PDFs!"
    tokens = tokenize(text)
    assert "enterprise" in tokens
    assert "ai" in tokens
    assert "rag" in tokens
    assert "pipelines" in tokens
    assert "chunk-size" in tokens or "chunk_size" in tokens or "chunk" in tokens


def test_tokenize_empty_and_none():
    """Verifies empty, whitespace, and None strings return empty list."""
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize(None) == []


def test_bm25_empty_query_returns_empty():
    """Verifies empty or whitespace queries return empty results without error."""
    chunks = [
        {"chunk_id": 1, "content": "Document indexing and search."},
    ]
    retriever = BM25Retriever.from_chunks(chunks)
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("nonexistentterm12345") == []


def test_bm25_exact_keyword_match():
    """Verifies exact keyword matches are ranked properly."""
    chunks = [
        {"chunk_id": 1, "content": "PostgreSQL database stores relational records."},
        {"chunk_id": 2, "content": "BM25 lexical search scores terms based on TF-IDF."},
        {"chunk_id": 3, "content": "Kubernetes container orchestration and cluster scaling."},
    ]
    retriever = BM25Retriever.from_chunks(chunks)

    results = retriever.search("BM25 lexical search", top_k=3)
    assert len(results) >= 1
    top = results[0]
    assert top["chunk_id"] == 2
    assert "BM25 lexical search" in top["content"]
    assert top["rank"] == 1
    assert top["source"] == "bm25"
    assert top["score"] > 0.0


def test_bm25_rare_term_influence():
    """Verifies rare terms across the corpus produce higher IDF and drive ranking."""
    chunks = [
        {"chunk_id": 1, "content": "The system processes enterprise documents every day."},
        {"chunk_id": 2, "content": "The system processes financial invoices and quarterly reports."},
        {"chunk_id": 3, "content": "The system processes compliance audits with strict cryptographic encryption."},
    ]
    retriever = BM25Retriever.from_chunks(chunks)

    # 'cryptographic' is rare (only in doc 3), whereas 'system' is common (in all 3)
    results = retriever.search("system cryptographic", top_k=3)
    assert len(results) >= 1
    # Chunk 3 must rank #1 because 'cryptographic' has significantly higher IDF
    assert results[0]["chunk_id"] == 3


def test_bm25_top_k_limiting():
    """Verifies top_k limits the returned result count."""
    chunks = [
        {"chunk_id": i, "content": f"Document number {i} discussing enterprise AI policies."}
        for i in range(1, 10)
    ]
    retriever = BM25Retriever.from_chunks(chunks)

    results = retriever.search("enterprise AI", top_k=3)
    assert len(results) == 3
    assert [r["rank"] for r in results] == [1, 2, 3]

    # top_k <= 0 should return empty list
    assert retriever.search("enterprise", top_k=0) == []
    assert retriever.search("enterprise", top_k=-2) == []


def test_bm25_metadata_preservation():
    """Verifies chunk metadata (page number, section, document name) is intact in results."""
    chunks = [
        {
            "chunk_id": "chunk_sec_01",
            "document_id": "doc_arch_001",
            "content": "Recursive structural chunking preserves section headings.",
            "metadata": {
                "document_name": "architecture.pdf",
                "page_number": 4,
                "section": "Chunking Rules",
                "chunk_index": 2,
            },
        }
    ]
    retriever = BM25Retriever.from_chunks(chunks)
    results = retriever.search("chunking section headings", top_k=1)

    assert len(results) == 1
    res = results[0]
    assert res["chunk_id"] == "chunk_sec_01"
    assert res["document_id"] == "doc_arch_001"
    assert res["metadata"]["document_name"] == "architecture.pdf"
    assert res["metadata"]["page_number"] == 4
    assert res["metadata"]["section"] == "Chunking Rules"
    assert res["metadata"]["chunk_index"] == 2


def test_bm25_from_file_factory():
    """Verifies BM25Retriever.from_file initializes from processed json files."""
    retriever = BM25Retriever.from_file()
    assert retriever.corpus_size > 0
    results = retriever.search("chunking overlap", top_k=2)
    assert len(results) > 0
    assert results[0]["score"] > 0.0


def test_search_bm25_convenience_function():
    """Verifies the standalone search_bm25 function works seamlessly with and without access context."""
    from app.models import AccessContext

    # Public corpus retrieval
    results = search_bm25("faq chunking", top_k=2)
    assert len(results) > 0
    assert all(r["source"] == "bm25" for r in results)

    # Manager-restricted corpus retrieval
    mgr_ctx = AccessContext(department="engineering", access_level="manager")
    mgr_results = search_bm25("governance policy", top_k=2, access_context=mgr_ctx)
    assert len(mgr_results) > 0
    assert all(r["source"] == "bm25" for r in mgr_results)

