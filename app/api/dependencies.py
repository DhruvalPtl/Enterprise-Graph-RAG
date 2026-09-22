"""
FastAPI Dependency Injection & Shared Lifecycle Management for Enterprise RAG (Step 7).

Manages the singleton lifecycle of the RAGPipeline:
1. Prevents recreating expensive ML models (MiniLM, Cross-Encoder) per HTTP request.
2. Reuses in-memory BM25 index and database connection configurations.
3. Supports dependency injection and test mocking via set_rag_pipeline.
"""
from typing import Optional
import logging

from app.rag import RAGPipeline

logger = logging.getLogger("rag_api")

# Module-level singleton reference
_shared_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """
    FastAPI dependency that returns the shared RAGPipeline singleton.
    Initializes the pipeline on first access if not already warmed up during startup.
    """
    global _shared_pipeline
    if _shared_pipeline is None:
        logger.info("Initializing shared RAGPipeline instance...")
        _shared_pipeline = RAGPipeline()
        logger.info("Shared RAGPipeline initialized successfully.")
    return _shared_pipeline


def set_rag_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    """
    Overrides the shared RAGPipeline singleton.
    Essential for unit tests to inject mock pipelines without loading live ML weights.
    """
    global _shared_pipeline
    _shared_pipeline = pipeline


def reset_rag_pipeline() -> None:
    """Resets the shared pipeline singleton back to None."""
    global _shared_pipeline
    _shared_pipeline = None
