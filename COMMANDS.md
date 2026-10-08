# Enterprise Knowledge Intelligence Platform — Command Reference Guide

This document provides a comprehensive list of all operational, CLI, Docker, database, API, and test commands for the Enterprise RAG system.

---

## Table of Contents

1. [Quick Start (Docker)](#1-quick-start-docker)
2. [Local Development Setup](#2-local-development-setup)
3. [Data Ingestion & Embedding Pipeline](#3-data-ingestion--embedding-pipeline)
4. [Search & Retrieval CLI Tools](#4-search--retrieval-cli-tools)
5. [Running the REST API Server](#5-running-the-rest-api-server)
6. [Testing the REST API (curl & PowerShell)](#6-testing-the-rest-api-curl--powershell)
7. [Automated Test Suite (pytest)](#7-automated-test-suite-pytest)
8. [Docker Operations & Maintenance](#8-docker-operations--maintenance)
9. [PostgreSQL & pgvector Direct Inspection](#9-postgresql--pgvector-direct-inspection)
10. [Troubleshooting & Quota Management](#10-troubleshooting--quota-management)

---

## 1. Quick Start (Docker)

The fastest way to spin up the complete platform (PostgreSQL with `pgvector` + FastAPI REST API with pre-loaded models):

```bash
# 1. Start all containers in detached mode
docker compose up -d

# 2. Check container health status
docker compose ps

# 3. Test API health
curl http://localhost:8000/health
```

---

## 2. Local Development Setup

### 2.1. Create and Activate Python Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3. Environment Configuration

Copy the example `.env` file and configure your Google Gemini API key:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_LLM_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=sentence-transformers
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=enterprise_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

---

## 3. Data Ingestion & Embedding Pipeline

Run these commands sequentially when populating or updating the enterprise knowledge base.

### 3.1. Generate Sample Benchmark Documents
Generates multi-page architectural PDFs and governance policies:
```bash
python scripts/generate_sample_pdf.py
```

### 3.2. Run Document Ingestion & Structure-Aware Chunking
Reads PDF, Markdown, and text files from `data/raw/` and produces deterministic chunks in `data/processed/all_chunks.json`:
```bash
python run_pipeline.py
```

### 3.3. Compute Dense Vector Embeddings
Generates 384-dimensional dense vector embeddings using local `sentence-transformers/all-MiniLM-L6-v2` and saves to `data/processed/embedded_chunks.json`:
```bash
python scripts/embed_chunks.py
```

### 3.4. Initialize PostgreSQL Schema & HNSW Vector Index
Connects to PostgreSQL, enables the `vector` extension, and builds tables (`documents`, `document_chunks`) with HNSW cosine distance indexing:
```bash
python scripts/init_db.py
```

### 3.5. Seed Chunks & Embeddings into PostgreSQL
Loads embedded chunks into PostgreSQL with department, access level, and active status metadata:
```bash
python scripts/store_chunks_in_db.py
```

---

## 4. Search & Retrieval CLI Tools

Use these CLI scripts to test and inspect each layer of the search and reasoning pipeline.

### 4.1. Dense Vector Similarity Search
Retrieves semantically similar chunks using cosine distance vector queries against pgvector:
```bash
# Default query
python scripts/search_vectors.py

# Custom query and top-k
python scripts/search_vectors.py --query "What is chunking and chunk overlap strategy?" --top-k 5
```

### 4.2. Hybrid Retrieval with Reciprocal Rank Fusion
Runs BM25 lexical search and pgvector semantic search in parallel and combines them via RRF ($k=60$):
```bash
# Default hybrid query
python scripts/search_hybrid.py

# Custom hybrid query with top-k limit
python scripts/search_hybrid.py --query "PostgreSQL vector index HNSW" --top-k 5
```

### 4.3. Two-Stage Retrieval with Cross-Encoder Reranking
Retrieves high-recall candidates via Hybrid RRF (Stage 1), then re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2` (Stage 2):
```bash
# Default two-stage search
python scripts/search_reranked.py

# Custom candidates pool (candidate-k) and final top-k
python scripts/search_reranked.py --query "What metadata is preserved during chunking?" --candidate-k 10 --top-k 3
```

### 4.4. Full End-to-End Grounded RAG Generation
Executes the full pipeline: Two-stage retrieval $\rightarrow$ Context builder $\rightarrow$ Grounded Gemini answer with strict citation attribution:
```bash
# General inquiry
python scripts/ask_rag.py --query "What is the primary purpose of the RAG pipeline?"

# Architecture inquiry
python scripts/ask_rag.py --query "What are the core architecture layers?" --top-k 3

# Out-of-corpus question (testing anti-hallucination refusal)
python scripts/ask_rag.py --query "What is the capital of Mars?"
```

---

## 5. Running the REST API Server

### 5.1. Run Locally (Uvicorn Dev Server)
Requires PostgreSQL running (e.g., via `docker compose up -d postgres`):
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5.2. Run via Docker Compose (Production)
```bash
docker compose up -d api
```

### 5.3. Accessing Interactive Documentation
Open in your browser:
- **Swagger UI Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Alternative Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### 5.4. Running the Interactive Streamlit Web UI
In a separate terminal with virtual environment activated:
```bash
streamlit run ui/app.py --server.port 8501
```
Open in your browser:
- **Streamlit Web Interface**: [http://localhost:8501](http://localhost:8501)


---

## 6. Testing the REST API (curl & PowerShell)

### 6.1. Health Check (`GET /health`)

**cURL (Linux / macOS / Git Bash):**
```bash
curl -s http://localhost:8000/health | jq
```

**PowerShell (Windows):**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json
```

---

### 6.2. Public RAG Query (`POST /query`)

**cURL:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the primary purpose of the RAG pipeline?",
    "top_k": 3
  }'
```

**PowerShell:**
```powershell
$body = @{
    query = "What is the primary purpose of the RAG pipeline?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

---

### 6.3. Access-Aware Query with Clearance Filtering (`POST /query`)

Demonstrates multi-tenant access control (`access_context`):

**Query restricted engineering architecture document (Employee Clearance):**
```powershell
$body = @{
    query = "What are the core architecture layers?"
    top_k = 3
    access_context = @{
        department = "engineering"
        access_level = "employee"
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

**Query privileged governance policy (Manager Clearance):**
```powershell
$body = @{
    query = "What are the AI model deployment requirements?"
    top_k = 3
    access_context = @{
        department = "engineering"
        access_level = "manager"
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

---

### 6.4. Out-of-Corpus Query (Hallucination Refusal Verification)

```powershell
$body = @{
    query = "What is the stock price of Apple in 2045?"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 3
```

---

### 6.5. Run Built-in Automated API Verifier

Runs the end-to-end verification script against all endpoints:
```bash
python scripts/verify_api.py
```

---

## 7. Automated Test Suite (pytest)

### 7.1. Run Entire Test Suite (All 126+ Tests)
```bash
pytest -v
```

### 7.2. Run Tests by Feature / Stage

| Stage / Component | Command |
|---|---|
| **Ingestion & Loaders** | `pytest tests/test_loaders.py tests/test_pipeline.py -v` |
| **SentenceTransformer Embeddings** | `pytest tests/test_embeddings.py -v` |
| **Database & Vector Search** | `pytest tests/test_db.py tests/test_vector_store.py -v` |
| **BM25 Lexical & Hybrid RRF** | `pytest tests/test_bm25.py tests/test_rrf.py tests/test_hybrid.py -v` |
| **Cross-Encoder Reranker** | `pytest tests/test_reranker.py -v` |
| **Grounded LLM & RAG Pipeline** | `pytest tests/test_llm.py tests/test_rag.py -v` |
| **FastAPI REST API & Schemas** | `pytest tests/test_api.py -v` |
| **Docker Packaging & Configuration** | `pytest tests/test_docker_config.py -v` |
| **Metadata Filtering & Access Control** | `pytest tests/test_access_control.py -v` |

### 7.3. Run Fast Unit Tests Only (Skipping Live Model / DB Calls)
```bash
pytest -m "not integration" -v
```

---

## 8. Docker Operations & Maintenance

### 8.1. Build API Container Image
```bash
# Build without cache
docker compose build --no-cache api

# Standard build
docker compose build api
```

### 8.2. Lifecycle Management
```bash
# Start all containers in background
docker compose up -d

# Start only PostgreSQL database container
docker compose up -d postgres

# Restart API container (e.g. after modifying .env)
docker compose restart api

# View real-time logs
docker compose logs -f api
docker compose logs -f postgres

# Stop all containers (preserves database data volume)
docker compose down

# Stop containers AND delete all volumes (factory reset)
docker compose down -v
```

### 8.3. Run Commands Inside Running Containers
```bash
# Run pytest inside the container
docker compose exec api pytest -v

# Run verification script inside container
docker compose exec api python scripts/verify_api.py

# Re-seed database inside container
docker compose exec api python scripts/store_chunks_in_db.py
```

---

## 9. PostgreSQL & pgvector Direct Inspection

Connect to PostgreSQL via `docker compose exec`:

```bash
docker compose exec postgres psql -U postgres -d enterprise_rag
```

Useful psql inspection queries:

```sql
-- Check installed extensions (verifies pgvector)
\dx

-- Count ingested documents and chunks
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM document_chunks;

-- Inspect stored chunks and their access control metadata
SELECT chunk_id, document_name, department, access_level, status
FROM document_chunks
LIMIT 10;

-- Verify vector dimension (should be 384)
SELECT chunk_id, vector_dims(embedding)
FROM document_chunks
LIMIT 1;

-- Test live cosine distance vector query manually
SELECT chunk_id, document_name, section,
       embedding <=> (SELECT embedding FROM document_chunks WHERE chunk_id = 50) AS cosine_dist
FROM document_chunks
ORDER BY cosine_dist ASC
LIMIT 3;

-- Exit psql
\q
```

---

## 10. Troubleshooting & Quota Management

### 10.1. Gemini API Quota Limit (`429 RESOURCE_EXHAUSTED`)
If you see `429 RESOURCE_EXHAUSTED (limit: 20, model: gemini-3.8-flash)`:
- Set `GEMINI_LLM_MODEL=gemini-3.5-flash-lite` in `.env`.
- Restart container:
  ```bash
  docker compose up -d --build api
  ```
- Flash-lite models have significantly higher free-tier request limits (1,500 requests/day).

### 10.2. Port Conflict on 5432 or 8000
If port `5432` is already used by a local PostgreSQL service:
- Stop local service: `net stop postgresql-x64-16` (Windows)
- Or edit `docker-compose.yml` to map host port: `"5433:5432"`.
