"""
Enterprise RAG API Client.

Provides synchronous HTTP client helpers using `httpx` for querying the
FastAPI backend service and checking system health.
"""
from typing import Dict, Any, Optional
import httpx


def check_health(api_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Check the liveness and database connectivity of the FastAPI backend.

    Args:
        api_url: Base URL of the API (e.g. 'http://localhost:8000').
        timeout: Maximum seconds to wait for a health response.

    Returns:
        Dict with status, database state, and error message if unreachable.
    """
    url = f"{api_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "ok": True,
                    "status": data.get("status", "ok"),
                    "database": data.get("database", "unknown"),
                    "version": data.get("version", "1.0.0"),
                    "app": data.get("app", "Enterprise RAG"),
                    "status_code": resp.status_code,
                }
            else:
                return {
                    "ok": False,
                    "status": f"HTTP {resp.status_code}",
                    "database": "unknown",
                    "error": f"Unexpected status code: {resp.status_code}",
                    "status_code": resp.status_code,
                }
    except httpx.ConnectError:
        return {
            "ok": False,
            "status": "offline",
            "database": "disconnected",
            "error": f"Cannot connect to backend at {api_url}. Is FastAPI running?",
        }
    except httpx.TimeoutException:
        return {
            "ok": False,
            "status": "timeout",
            "database": "unknown",
            "error": f"Backend at {api_url} timed out after {timeout}s.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "database": "unknown",
            "error": str(exc),
        }


def submit_query(
    api_url: str,
    query: str,
    top_k: int = 5,
    candidate_k: Optional[int] = None,
    retrieval_mode: str = "hybrid_graph_vector",
    access_context: Optional[Dict[str, Any]] = None,
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """
    Send a natural language search query to the RAG backend POST /query endpoint.

    Args:
        api_url: Base URL of the API.
        query: User's question or search phrase.
        top_k: Number of final evidence citations to return (1..50).
        candidate_k: Optional hybrid candidate pool size (1..100).
        retrieval_mode: 'hybrid_graph_vector' or 'vector_bm25_rrf'.
        access_context: Caller clearance dict (department, access_level, include_archived).
        timeout: Maximum seconds to wait for inference and generation.

    Returns:
        Dict with 'ok' boolean and either 'data' (QueryResponse) or error details.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return {
            "ok": False,
            "error_type": "client_validation",
            "detail": "Query cannot be empty or only whitespace.",
        }

    payload: Dict[str, Any] = {
        "query": cleaned_query,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
    }
    if candidate_k is not None:
        payload["candidate_k"] = candidate_k
    if access_context is not None:
        payload["access_context"] = access_context

    url = f"{api_url.rstrip('/')}/query"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)

            if resp.status_code == 200:
                return {
                    "ok": True,
                    "data": resp.json(),
                    "status_code": 200,
                }
            elif resp.status_code == 422:
                # Validation error from FastAPI / Pydantic
                return {
                    "ok": False,
                    "error_type": "validation_error",
                    "status_code": 422,
                    "detail": resp.json().get("detail", resp.text),
                }
            else:
                return {
                    "ok": False,
                    "error_type": "server_error",
                    "status_code": resp.status_code,
                    "detail": resp.text,
                }
    except httpx.ConnectError:
        return {
            "ok": False,
            "error_type": "connection_error",
            "detail": f"Failed to connect to RAG API at {api_url}. Please verify the server is running.",
        }
    except httpx.TimeoutException:
        return {
            "ok": False,
            "error_type": "timeout",
            "detail": f"Request to {url} timed out after {timeout:.0f} seconds.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": "unknown",
            "detail": str(exc),
        }
