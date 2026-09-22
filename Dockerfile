# ==============================================================================
# Enterprise Knowledge Intelligence Platform - Production Dockerfile (Step 8)
# ==============================================================================
# Base image: Official Python 3.12 slim Debian (Bookworm)
FROM python:3.12-slim-bookworm

# Set Python environment variables for container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install minimal essential system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root application user and cache directories
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser && \
    mkdir -p /app /app/.cache/huggingface && \
    chown -R appuser:appuser /app

# Install Python dependencies separately to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code, scripts, and processed data
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser data/ /app/data/

# Switch to unprivileged application user
USER appuser

# Expose FastAPI REST API port
EXPOSE 8000

# Container healthcheck using lightweight /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI production application with Uvicorn
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
