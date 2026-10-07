"""
FastAPI REST API Application for Enterprise Knowledge Intelligence Platform (Step 7).

Endpoints:
- GET  /health : Liveness check and database readiness status.
- POST /query  : End-to-end question-answering via RAGPipeline with validated schemas.
- GET  /docs   : OpenAPI interactive Swagger UI documentation.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
)
from app.db import test_connection, init_db
from app.llm import sanitize_error_message
from app.models import AccessContext
from app.rag import RAGPipeline
from app.api.schemas import QueryRequest, QueryResponse, HealthResponse
from app.api.dependencies import get_rag_pipeline, set_rag_pipeline, reset_rag_pipeline

# Configure server logger with updated graph retriever lifecycle
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan event handler for FastAPI.
    1. Verifies/initializes the PostgreSQL schema and pgvector extension idempotently.
    2. Warms up shared models and resources during startup to prevent per-request latency spikes.
    """
    logger.info(f"Starting {API_TITLE} v{API_VERSION}...")
    try:
        # Check and initialize database schema if database is available
        if test_connection():
            logger.info("PostgreSQL connection confirmed. Verifying database schema & pgvector extension...")
            init_db()
            logger.info("Database schema & pgvector extension verified.")
        else:
            logger.warning("PostgreSQL not immediately reachable during startup; schema initialization deferred.")

        # Pre-warm shared RAG pipeline
        get_rag_pipeline()
        logger.info("RAG pipeline pre-warmed and ready for requests.")
    except Exception as exc:
        logger.warning(
            f"Note: RAG pipeline warm-up deferred or partially initialized: {sanitize_error_message(str(exc))}"
        )
    yield
    logger.info("Shutting down API server and releasing resources...")
    reset_rag_pipeline()


# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# -------------------------------------------------------------------------
# Custom Exception Handlers (Security & Sanitization)
# -------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors with clean HTTP 422 responses."""
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field_loc,
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": errors,
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles business-logic validation errors with HTTP 400."""
    clean_msg = sanitize_error_message(str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": clean_msg},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled server exceptions.
    Prevents leaking internal stack traces, API keys, or database credentials.
    """
    clean_msg = sanitize_error_message(str(exc))
    logger.error(f"Internal server error processing {request.url.path}: {clean_msg}", exc_info=True)

    # Differentiate upstream provider failures (502) from internal errors (500)
    if "Gemini generation failed" in clean_msg or "UNAVAILABLE" in clean_msg:
        status_code = status.HTTP_502_BAD_GATEWAY
        client_msg = "Upstream LLM provider is temporarily unavailable. Please try again."
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        client_msg = "An internal server error occurred while processing the request."

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": client_msg,
            "error_type": exc.__class__.__name__,
        },
    )


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check & database readiness",
)
def health_check() -> HealthResponse:
    """
    Returns the operational status of the API and its underlying PostgreSQL database.
    Does not call expensive ML models or external LLMs.
    """
    db_ok = test_connection()
    db_status = "connected" if db_ok else "disconnected"
    overall_status = "ok" if db_ok else "degraded"

    return HealthResponse(
        status=overall_status,
        app=API_TITLE,
        version=API_VERSION,
        database=db_status,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
    summary="Ask a question to the enterprise RAG platform",
)
def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """
    Executes end-to-end question answering:
    1. Validates query text and bounds.
    2. Runs Stage 1 hybrid retrieval (Vector + BM25 -> RRF).
    3. Runs Stage 2 fine Cross-Encoder reranking.
    4. Formats whole-passage bounded context and compiles authoritative citations.
    5. Generates grounded answer via Gemini LLM (or returns structured refusal if no evidence).
    6. Returns structured response with answer and citations.
    """
    # Map API request access_context to domain model AccessContext
    # Default to strictly unprivileged public access if not provided
    if request.access_context:
        access_ctx = AccessContext(
            department=request.access_context.department,
            access_level=request.access_context.access_level,
            include_archived=request.access_context.include_archived,
        )
    else:
        access_ctx = AccessContext(department="public", access_level="public", include_archived=False)

    try:
        rag_response = pipeline.answer_query(
            query=request.query,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
            access_context=access_ctx,
        )
        return QueryResponse.from_rag_response(rag_response)
    except HTTPException:
        raise
    except Exception as exc:
        clean_msg = sanitize_error_message(str(exc))
        logger.error(f"Error during RAG query execution: {clean_msg}")
        if "Gemini generation failed" in clean_msg or "UNAVAILABLE" in clean_msg:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream LLM provider is temporarily unavailable. Please try again.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing the request.",
        )

