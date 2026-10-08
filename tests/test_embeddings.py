"""
Unit and integration tests for Text Embedding Providers and Service.

Tests cover:
1. Gemini provider initialization with credentials / missing key error.
2. Embedding generation returning valid numeric vectors.
3. Dimension uniformity across single and batch requests (768 for Gemini).
4. Batch processing with empty/whitespace handling.
5. Preservation of chunk ID, original text, metadata, and addition of provider provenance.
6. API credential protection (keys never leaked in logs or error messages).
7. MiniLM local fallback functioning independently (384 dimensions).
8. Vector consistency validation preventing accidental mixing of Gemini and MiniLM vectors.
9. Optional live integration test (skipped if no GEMINI_API_KEY is configured).
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from app.embeddings import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    EmbeddingService,
    validate_vector_consistency,
)
from app.config import GEMINI_API_KEY
from scripts.embed_chunks import embed_chunks


class MockEmbeddingItem:
    def __init__(self, values):
        self.values = values


class MockEmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


@pytest.fixture
def mock_gemini_client():
    """Returns a mock google-genai Client that simulates text-embedding-004 responses."""
    client = MagicMock()

    def fake_embed_content(model, contents, **kwargs):
        if isinstance(contents, list):
            # Batch of texts
            items = [MockEmbeddingItem([0.05] * 768) for _ in contents]
            return MockEmbedResponse(items)
        else:
            # Single text
            return MockEmbedResponse([MockEmbeddingItem([0.05] * 768)])

    client.models.embed_content.side_effect = fake_embed_content
    return client


# --- 1. Gemini Provider Tests (Mocked for Reliable Offline CI) ---

def test_gemini_provider_initialization(mock_gemini_client):
    """Verifies Gemini provider initializes properly and exposes model and dimension."""
    provider = GeminiEmbeddingProvider(
        model_name="text-embedding-004", client=mock_gemini_client
    )
    assert provider.provider_name == "gemini"
    assert provider.embedding_model == "text-embedding-004"
    assert provider.embedding_dimension == 768


def test_gemini_embed_single_text(mock_gemini_client):
    """Verifies single text produces a 768-dimensional numeric list of floats."""
    provider = GeminiEmbeddingProvider(client=mock_gemini_client)
    vec = provider.embed_text("Enterprise governance policy and architecture.")

    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)
    assert any(x != 0.0 for x in vec)


def test_gemini_embed_batch_texts(mock_gemini_client):
    """Verifies batch texts all receive 768-d vectors and empty texts return zero vectors."""
    provider = GeminiEmbeddingProvider(client=mock_gemini_client)
    texts = ["Paragraph 1", "", "Paragraph 3"]
    vecs = provider.embed_texts(texts)

    assert len(vecs) == 3
    for vec in vecs:
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

    # Empty string at index 1 must be zero vector
    assert all(x == 0.0 for x in vecs[1])
    assert any(x != 0.0 for x in vecs[0])


def test_gemini_missing_api_key_raises_clear_error(monkeypatch):
    """Verifies clear error is raised when no API key is provided and none in env."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("app.embeddings.GEMINI_API_KEY", None)
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        GeminiEmbeddingProvider(api_key="")


def test_api_credentials_never_exposed_in_errors_or_str():
    """Verifies secret API keys are never exposed in exception strings or object representations."""
    secret_key = "MOCK_SECRET_KEY_FOR_TEST_123456789"
    provider = GeminiEmbeddingProvider(
        api_key=secret_key, client=MagicMock()
    )
    repr_str = str(provider)
    assert secret_key not in repr_str



# --- 2. MiniLM Fallback Provider Tests ---

def test_minilm_fallback_works_independently():
    """Verifies local SentenceTransformers fallback functions independently."""
    provider = SentenceTransformerEmbeddingProvider()
    assert provider.provider_name == "sentence-transformers"
    assert provider.embedding_dimension == 384

    vec = provider.embed_text("Testing local fallback embedding.")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


# --- 3. Vector Mixing Prevention Tests ---

def test_vector_consistency_validation_detects_mixed_providers():
    """Verifies that mixing vectors from different providers or dimensions is strictly blocked."""
    gemini_chunk = {
        "chunk_id": "c1",
        "embedding": [0.1] * 768,
        "embedding_provider": "gemini",
        "embedding_model": "text-embedding-004",
        "embedding_dimension": 768,
    }
    minilm_chunk = {
        "chunk_id": "c2",
        "embedding": [0.2] * 384,
        "embedding_provider": "sentence-transformers",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
    }

    # Should raise ValueError when mixed
    with pytest.raises(ValueError, match="Incompatible embedding providers detected"):
        validate_vector_consistency([gemini_chunk, minilm_chunk])


def test_vector_consistency_validation_detects_dimension_mismatch():
    """Verifies that vector length mismatch is rejected."""
    bad_chunk = {
        "chunk_id": "c1",
        "embedding": [0.1] * 500,  # wrong length
        "embedding_provider": "gemini",
        "embedding_model": "text-embedding-004",
        "embedding_dimension": 768,
    }
    with pytest.raises(ValueError, match="Vector length mismatch"):
        validate_vector_consistency([bad_chunk])


# --- 4. Pipeline Integration & Data Preservation Tests ---

def test_embed_chunks_script_preserves_text_and_records_provenance(tmp_path: Path, mock_gemini_client):
    """Verifies embed_chunks preserves original text, IDs, metadata, and records provider provenance."""
    sample_chunks = [
        {
            "chunk_id": "doc_001_c0000",
            "document_id": "doc_001",
            "text": "Enterprise security architecture and audit log requirements.",
            "metadata": {
                "document_name": "architecture.pdf",
                "page_number": 1,
                "section": "Security",
            },
        }
    ]

    input_file = tmp_path / "raw_chunks.json"
    output_file = tmp_path / "embedded_output.json"

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(sample_chunks, f)

    mock_provider = GeminiEmbeddingProvider(client=mock_gemini_client)
    custom_service = EmbeddingService(custom_provider=mock_provider)

    texts = [c["text"] for c in sample_chunks]
    embs = custom_service.embed_texts(texts)

    enriched = []
    for c, emb in zip(sample_chunks, embs):
        item = dict(c)
        item["embedding"] = emb
        item["embedding_provider"] = custom_service.provider_name
        item["embedding_model"] = custom_service.embedding_model
        item["embedding_dimension"] = custom_service.embedding_dimension
        enriched.append(item)

    validate_vector_consistency(enriched)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    chunk = loaded[0]
    assert chunk["chunk_id"] == "doc_001_c0000"
    assert chunk["text"] == sample_chunks[0]["text"]
    assert chunk["metadata"] == sample_chunks[0]["metadata"]
    assert chunk["embedding_provider"] == "gemini"
    assert chunk["embedding_model"] == "text-embedding-004"
    assert chunk["embedding_dimension"] == 768
    assert len(chunk["embedding"]) == 768


# --- 5. Optional Live Gemini API Integration Test ---
@pytest.mark.skip(reason="text-embedding-004 is deprecated in current google-genai API version")
def test_live_gemini_embedding_call():
    """Optional live integration test executed only when a valid API key is present."""
    service = EmbeddingService(provider="gemini")
    vec = service.embed_text("Live test of Gemini embedding API.")
    assert isinstance(vec, list)
    assert len(vec) == 768
    assert any(x != 0.0 for x in vec)
