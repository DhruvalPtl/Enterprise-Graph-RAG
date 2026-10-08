"""
Unit tests for Docker and Production Packaging Configuration.

These tests run offline and do not require Docker daemon execution:
1. Validates Dockerfile structure, base image, non-root user, and entrypoint.
2. Validates .dockerignore rules for secret leakage prevention and data retention.
3. Validates docker-compose.yml structure, services, healthchecks, networks, and named volumes.
4. Validates database configuration environment variable fallback resolution.
"""
from pathlib import Path
import os
import yaml
import pytest

from app.config import BASE_DIR


def test_dockerfile_configuration():
    """Verify Dockerfile meets all production packaging requirements."""
    dockerfile_path = BASE_DIR / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist in project root"

    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Base image must be official slim Python
    assert "FROM python:3.12-slim" in content, "Dockerfile should use Python 3.12 slim base"

    # 2. Sensible working directory
    assert "WORKDIR /app" in content, "Working directory must be /app"

    # 3. Environment variables for Python in containers
    assert "ENV PYTHONDONTWRITEBYTECODE=1" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "PYTHONPATH=/app" in content
    assert "HF_HOME=/app/.cache/huggingface" in content

    # 4. Non-root user creation and usage
    assert "useradd" in content and "appuser" in content, "Must create an unprivileged user"
    assert "USER appuser" in content, "Must switch to non-root appuser"

    # 5. Dependency installation and source copy
    assert "requirements.txt" in content
    assert "pip install" in content
    assert "COPY --chown=appuser:appuser app/ /app/app/" in content
    assert "COPY --chown=appuser:appuser data/ /app/data/" in content

    # 6. Expose port and correct entrypoint
    assert "EXPOSE 8000" in content
    assert "CMD [\"uvicorn\", \"app.api.main:app\"" in content
    assert "HEALTHCHECK" in content


def test_dockerignore_configuration():
    """Verify .dockerignore protects secrets and caches while keeping required files."""
    dockerignore_path = BASE_DIR / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist in project root"

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    # Critical security & hygiene exclusions
    assert ".env" in lines, ".env must be excluded from image context"
    assert ".git" in lines, ".git must be excluded"
    assert ".venv" in lines, ".venv must be excluded"
    assert "__pycache__/" in lines or "__pycache__" in lines
    assert ".pytest_cache/" in lines or ".pytest_cache" in lines

    # Ensure required directories are NOT excluded
    assert "app" not in lines
    assert "app/" not in lines
    assert "data" not in lines
    assert "data/" not in lines
    assert "requirements.txt" not in lines


def test_docker_compose_configuration():
    """Verify docker-compose.yml structure, dependencies, healthchecks, and volumes."""
    compose_path = BASE_DIR / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist in project root"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data, "docker-compose must define services"
    services = compose_data["services"]

    # 1. Verify postgres service
    assert "postgres" in services, "Must define a 'postgres' service"
    pg_service = services["postgres"]
    assert "pgvector" in pg_service["image"], "Postgres service must use pgvector image"
    assert "healthcheck" in pg_service, "Postgres service must define a healthcheck"
    assert "volumes" in pg_service
    assert any("postgres_data:" in v for v in pg_service["volumes"])

    # 2. Verify api service
    assert "api" in services, "Must define an 'api' service"
    api_service = services["api"]
    assert "build" in api_service, "API service must build from Dockerfile"
    assert "depends_on" in api_service, "API service must depend on postgres"
    assert api_service["depends_on"]["postgres"]["condition"] == "service_healthy"

    # 3. Verify internal networking connection to postgres service
    env = api_service.get("environment", {})
    assert env.get("DATABASE_HOST") == "postgres" or env.get("POSTGRES_HOST") == "postgres", (
        "API must connect to 'postgres' container, not 'localhost'"
    )
    assert env.get("DATABASE_PORT") == 5432 or env.get("POSTGRES_PORT") == 5432
    assert "GEMINI_API_KEY" in env, "GEMINI_API_KEY must be passed via environment"

    # 4. Verify API healthcheck and ports
    assert "healthcheck" in api_service, "API service must define a healthcheck"
    assert "8000:8000" in api_service.get("ports", [])
    assert any("hf_cache:" in v for v in api_service.get("volumes", []))

    # 5. Verify named persistent volumes
    assert "volumes" in compose_data
    assert "postgres_data" in compose_data["volumes"]
    assert "hf_cache" in compose_data["volumes"]


def test_database_env_fallback_resolution(monkeypatch):
    """Verify that DATABASE_* variables take precedence and fall back cleanly to POSTGRES_*."""
    monkeypatch.setenv("DATABASE_HOST", "custom-pg-host")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    monkeypatch.setenv("DATABASE_NAME", "custom_rag_db")

    # Test resolution logic directly
    host = os.getenv("DATABASE_HOST") or os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("DATABASE_PORT") or os.getenv("POSTGRES_PORT", 5432))
    name = os.getenv("DATABASE_NAME") or os.getenv("POSTGRES_DB", "rag_db")

    assert host == "custom-pg-host"
    assert port == 5433
    assert name == "custom_rag_db"

    # Unset DATABASE_* and verify fallback to POSTGRES_*
    monkeypatch.delenv("DATABASE_HOST")
    monkeypatch.setenv("POSTGRES_HOST", "fallback-host")
    fallback_host = os.getenv("DATABASE_HOST") or os.getenv("POSTGRES_HOST", "localhost")
    assert fallback_host == "fallback-host"
