"""
Unit and Integration tests for Vector Store and Cosine Retrieval.

Tests cover:
- Dimension validation (strict 384-d enforcement, rejecting 768-d Gemini vectors)
- Empty query validation
- Mocked SQL query formulation and parameter verification
- VectorStore facade class functionality
- Live PostgreSQL integration tests for cosine distance ranking (<=>) and HNSW indexing
"""
import pytest
from unittest.mock import MagicMock, patch

from app.config import VECTOR_DIMENSION
from app.db import (
    test_connection as check_db_connection,
    init_db,
    get_connection,
    insert_document,
    insert_chunks,
)
from app.vector_store import (
    search_similar_chunks,
    retrieve_similar_chunks,
    VectorStore,
)


# --- 1. Unit Tests (Fast & Offline) ---

def test_search_similar_chunks_requires_non_none_vector():
    """Verifies ValueError is raised if query_embedding is None."""
    with pytest.raises(ValueError, match="query_embedding cannot be None"):
        search_similar_chunks(None)


def test_search_similar_chunks_rejects_incompatible_dimensions():
    """Verifies search_similar_chunks rejects non-384 vectors (such as 768-d Gemini embeddings)."""
    gemini_vector = [0.01] * 768
    with pytest.raises(ValueError) as exc_info:
        search_similar_chunks(gemini_vector)

    assert "Incompatible query vector dimension" in str(exc_info.value)
    assert "384" in str(exc_info.value)
    assert "768" in str(exc_info.value)

    short_vector = [0.1] * 128
    with pytest.raises(ValueError) as exc_info:
        search_similar_chunks(short_vector)
    assert "128" in str(exc_info.value)


def test_search_similar_chunks_top_k_zero_or_negative():
    """Verifies top_k <= 0 safely returns an empty list without hitting database."""
    dummy_vec = [0.0] * VECTOR_DIMENSION
    assert search_similar_chunks(dummy_vec, top_k=0) == []
    assert search_similar_chunks(dummy_vec, top_k=-5) == []


def test_retrieve_similar_chunks_empty_query_raises_error():
    """Verifies retrieve_similar_chunks rejects empty or whitespace-only queries."""
    with pytest.raises(ValueError, match="query_text cannot be empty"):
        retrieve_similar_chunks("")

    with pytest.raises(ValueError, match="query_text cannot be empty"):
        retrieve_similar_chunks("   \n\t  ")


def test_search_similar_chunks_mocked():
    """Verifies SQL formulation and dictionary result formatting with a mock cursor."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    mock_cur.fetchall.return_value = [
        {
            "chunk_id": 1,
            "document_id": 10,
            "content": "Sample chunk content",
            "page_number": 2,
            "section": "Architecture",
            "chunk_index": 0,
            "document_name": "arch.pdf",
            "document_type": "pdf",
            "distance": 0.15,
        }
    ]

    dummy_vec = [0.05] * VECTOR_DIMENSION
    results = search_similar_chunks(dummy_vec, top_k=1, conn=mock_conn)

    assert len(results) == 1
    res = results[0]
    assert res["chunk_id"] == 1
    assert res["document_id"] == 10
    assert res["content"] == "Sample chunk content"
    assert res["distance"] == 0.15
    assert res["similarity"] == 0.85
    assert res["metadata"]["document_name"] == "arch.pdf"
    assert res["metadata"]["page_number"] == 2

    # Verify SQL query used cosine distance operator (<=>)
    call_args = mock_cur.execute.call_args
    sql_text = call_args[0][0]
    assert "<=>" in sql_text
    assert "ORDER BY distance ASC" in sql_text


def test_retrieve_similar_chunks_delegates_to_embedding_service():
    """Verifies retrieve_similar_chunks calls embed_text and passes vector to search."""
    mock_service = MagicMock()
    dummy_vec = [0.1] * VECTOR_DIMENSION
    mock_service.embed_text.return_value = dummy_vec

    with patch("app.vector_store.search_similar_chunks") as mock_search:
        mock_search.return_value = [{"chunk_id": 99}]
        results = retrieve_similar_chunks(
            query_text="Explain chunking",
            top_k=3,
            embedding_service=mock_service,
        )

        mock_service.embed_text.assert_called_once_with("Explain chunking")
        mock_search.assert_called_once_with(query_embedding=dummy_vec, top_k=3, access_context=None, conn=None)
        assert results == [{"chunk_id": 99}]


def test_vector_store_facade_class():
    """Verifies VectorStore object-oriented class properly delegates search and retrieve."""
    mock_service = MagicMock()
    store = VectorStore(embedding_service=mock_service)

    dummy_vec = [0.2] * VECTOR_DIMENSION
    with patch("app.vector_store.search_similar_chunks") as mock_search:
        mock_search.return_value = []
        store.search(dummy_vec, top_k=5)
        mock_search.assert_called_once_with(query_embedding=dummy_vec, top_k=5, access_context=None, conn=None)

    with patch("app.vector_store.retrieve_similar_chunks") as mock_retrieve:
        mock_retrieve.return_value = []
        store.retrieve("What is RAG?", top_k=2)
        mock_retrieve.assert_called_once_with(
            query_text="What is RAG?",
            top_k=2,
            access_context=None,
            conn=None,
            embedding_service=mock_service,
        )


# --- 2. Live Integration Tests against PostgreSQL with pgvector ---

IS_POSTGRES_AVAILABLE = check_db_connection()


@pytest.mark.skipif(
    not IS_POSTGRES_AVAILABLE,
    reason="Live PostgreSQL instance not detected on localhost:5432. Integration tests skipped.",
)
class TestLiveVectorSearchIntegration:
    """
    Live integration tests for pgvector cosine similarity search and HNSW indexing.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Prepares database and cleans up test data after each test."""
        init_db()
        conn = get_connection(autocommit=True)
        yield conn
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE filename LIKE 'test_vec_%';")
        conn.close()

    def test_live_cosine_similarity_ranking(self, setup_and_teardown):
        """
        Inserts known synthetic vectors and verifies pgvector cosine distance ranking.
        - Vector A is aligned with Query Vector (cosine distance = 0.0)
        - Vector B is orthogonal to Query Vector (cosine distance = 1.0)
        """
        conn = setup_and_teardown
        doc_id = insert_document(conn, filename="test_vec_doc.md", document_type="markdown")

        # Query vector: [1.0, 0.0, 0.0, ... 0.0]
        query_vec = [0.0] * VECTOR_DIMENSION
        query_vec[0] = 1.0

        # Chunk A vector: identical to query
        vec_a = list(query_vec)

        # Chunk B vector: orthogonal to query [0.0, 1.0, 0.0, ... 0.0]
        vec_b = [0.0] * VECTOR_DIMENSION
        vec_b[1] = 1.0

        test_chunks = [
            {"text": "Chunk B (Orthogonal)", "metadata": {"chunk_index": 0}, "embedding": vec_b},
            {"text": "Chunk A (Identical)", "metadata": {"chunk_index": 1}, "embedding": vec_a},
        ]
        insert_chunks(conn, document_id=doc_id, chunks=test_chunks)

        # Execute search
        results = search_similar_chunks(query_embedding=query_vec, top_k=5, conn=conn)

        # Chunk A must rank first for query_vec with distance ~ 0.0 and similarity ~ 1.0
        assert len(results) >= 1
        first_match = results[0]
        assert first_match["content"] == "Chunk A (Identical)"
        assert first_match["distance"] == pytest.approx(0.0, abs=1e-3)
        assert first_match["similarity"] == pytest.approx(1.0, abs=1e-3)

        # Querying with vec_b must rank Chunk B first with distance ~ 0.0 and similarity ~ 1.0
        b_results = search_similar_chunks(query_embedding=vec_b, top_k=5, conn=conn)
        assert len(b_results) >= 1
        b_match = b_results[0]
        assert b_match["content"] == "Chunk B (Orthogonal)"
        assert b_match["distance"] == pytest.approx(0.0, abs=1e-3)
        assert b_match["similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_live_end_to_end_retrieve_similar_chunks(self, setup_and_teardown):
        """Verifies end-to-end retrieve_similar_chunks using MiniLM text embedding."""
        conn = setup_and_teardown
        doc_id = insert_document(conn, filename="test_vec_policy.md", document_type="markdown")

        # Create embedding for a specific policy chunk
        from app.embeddings import EmbeddingService
        service = EmbeddingService(provider="sentence-transformers")

        policy_text = "All engineers must obtain architectural sign-off before deploying RAG pipelines."
        emb = service.embed_text(policy_text)

        insert_chunks(
            conn,
            document_id=doc_id,
            chunks=[{
                "text": policy_text,
                "metadata": {"section": "Deployment", "page_number": 1, "chunk_index": 0},
                "embedding": emb,
            }],
        )

        # Search for semantically related question
        query = "Who needs architectural approval for RAG deployment?"
        results = retrieve_similar_chunks(query_text=query, top_k=1, conn=conn, embedding_service=service)

        assert len(results) >= 1
        top = results[0]
        assert policy_text in top["content"]
        assert top["similarity"] > 0.5
        assert top["distance"] < 0.5
        assert top["metadata"]["section"] == "Deployment"
