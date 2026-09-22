# Project Audit Report: Enterprise RAG & Graph RAG (Phase G6)

- **Audit Date**: 2026-09-22
- **Project Root**: `D:\Programing\RAG`
- **Target Audience**: Forward Deployed AI Engineer / Enterprise Open-Source Publication
- **Status**: Audit Complete | Zero Destructive Changes

---

## 1. Executive Summary

This comprehensive audit inventories and categorizes all directories, modules, scripts, tests, datasets, Docker configurations, evaluation artifacts, and caches across the repository. The objective is to prepare the project for public GitHub release and upcoming UI integration while enforcing strict security, operational safety, and empirical honesty.

---

## 2. Directory Inventory & Module Architecture

| Directory | Purpose / Category | File Count | Size on Disk |
| :--- | :--- | :--- | :--- |
| `app/` | Core Python application modules (API, RAG, Graph, Evaluation) | 26 files | ~230 KB |
| `app/api/` | FastAPI REST server, dependencies, and schemas | 4 files | ~16 KB |
| `app/evaluation/` | G5 Evaluation benchmark runner, metrics, and dataset loader | 5 files | ~46 KB |
| `scripts/` | Operational, extraction, migration, and benchmarking scripts | 18 files | ~170 KB |
| `scripts/benchmark/` | OCR and extraction analysis scripts | 3 files | ~32 KB |
| `tests/` | Complete unit and integration test suite | 26 files | ~240 KB |
| `data/raw/` | Primary source documents (Stanford AI Index 2024, Jurafsky & Martin NLP) | 4 files | ~26.1 MB |
| `data/processed/` | Intermediate chunk JSONs and vector embeddings | 4 files | ~82.7 MB |
| `data/evaluation/` | G5 gold-standard evaluation dataset and benchmark traces | 4 files | ~175 KB |
| `data/archived_books/` | Consolidate textbook PDFs and historical embeddings | 8 files | ~185.2 MB |
| `reports/` | Empirical evaluation reports, extraction audits, and benchmarks | 10 files | ~2.2 MB |
| `reports/ocr_benchmark/` | OCR comparison analysis and sample cropped images | 21 files | ~3.6 MB |
| `learn/` | Scratch learning scripts (hash, test chunker, embedding) | 3 files | ~2.5 KB |
| Root Files | Docker, config, requirements, environment, documentation | 12 files | ~140 KB |

---

## 3. Comprehensive File Classification

Files are classified into 5 strict operational categories:
- **A. MUST KEEP**: Core production code, tests, schemas, evaluation dataset, primary documentation.
- **B. SHOULD KEEP BUT NOT COMMIT**: Large files (>10 MB), local database checkpoints, embedding caches, credentials (tracked in `.gitignore`).
- **C. SAFE TO REMOVE**: Ephemeral caches (`__pycache__`, `.pytest_cache`, temporary logs).
- **D. NEEDS REVIEW**: Early experimental scripts and historical utilities.
- **E. POTENTIAL SECRET / SECURITY RISK**: Files containing live API keys or passwords.

### Category E: Potential Secret / Security Risk
| File Path | Risk Type | Recommended Action |
| :--- | :--- | :--- |
| `api_key.text` | Contains 4 active Google Gemini API keys | **DO NOT COMMIT**. Add to `.gitignore` and `.dockerignore`. Keep locally only. |
| `.env` | Contains PostgreSQL passwords and active runtime keys | **DO NOT COMMIT**. Add to `.gitignore`. Use `.env.example` as template. |

### Category B: Should Keep But Not Commit (Local-Only / Git LFS)
| File Path | Size | Reason / Classification |
| :--- | :--- | :--- |
| `data/archived_books/processed/embedded_chunks_full_11609.json` | 136.70 MB | **Exceeds GitHub 100 MB hard limit**. Local historical artifact. Add to `.gitignore`. |
| `data/processed/embedded_chunks.json` | 67.51 MB | Large pre-computed embedding cache. Can be regenerated. Add to `.gitignore`. |
| `data/archived_books/*.pdf` (3 files) | 37.0 MB total | Archived heavy textbook PDFs. Exclude from Git repository. |
| `data/graph_extraction_progress.db` | 864 KB | SQLite checkpoint database tracking live extraction. Add to `.gitignore`. |
| `data/archived_books/graph_extraction_progress_backup.db` | 1.14 MB | SQLite checkpoint backup. Add to `.gitignore`. |
| `reports/ocr_benchmark/*.png` (6 files) | ~3.5 MB | Diagnostic image crops from OCR comparison. Add to `.gitignore`. |
| `.venv/` | ~3.8 GB | Python virtual environment. Add to `.gitignore`. |

### Category C: Safe to Remove (Caches & Ephemera)
| File Pattern / Directory | Description |
| :--- | :--- |
| `**/__pycache__/` | Compiled Python bytecode (`.pyc`). |
| `.pytest_cache/` | Pytest session and cache directories. |
| Temporary log files | Any ephemeral `.log` files generated during benchmarking. |

### Category D: Needs Review (Historical / Experimental)
| File Path | Purpose | Recommendation |
| :--- | :--- | :--- |
| `learn/hash.py` | Early hashing test | Retain in repo as educational/exploratory reference. |
| `learn/test_chunker.py` | Early standalone chunker test | Retain in repo. |
| `learn/test_embedding.py` | Early standalone embedding test | Retain in repo. |
| `scripts/cleanup_corpus_for_fast_graph.py` | Consolidated corpus to 2 books | Retain for historical reproducibility. |
| `scripts/run_graph_extraction_prototype.py` | G2.1 extraction prototype | Retain for historical reference. |
| `scripts/generate_sample_pdf.py` | Generates `enterprise_platform_architecture.pdf` | **Crucial for unit tests**; keep in `scripts/`. |

### Category A: Must Keep (Repository Core)
- **Application Core (`app/`)**:
  - `app/api/`: `main.py`, `schemas.py`, `dependencies.py`
  - `app/evaluation/`: `dataset.py`, `metrics.py`, `runner.py`, `reporting.py`
  - `app/graph_*`: `graph_retriever.py`, `graph_ontology.py`, `graph_extractor.py`, `graph_checkpoint.py`
  - `app/hybrid*.py`: `hybrid_retriever.py`, `hybrid.py`, `rrf.py`
  - `app/rag.py`, `reranker.py`, `context_builder.py`, `llm.py`, `vector_store.py`, `bm25.py`, `embeddings.py`, `chunker.py`, `loaders.py`, `models.py`, `db.py`, `config.py`
- **Test Suite (`tests/`)**: All 26 test files.
- **Evaluation Dataset (`data/evaluation/`)**: `graph_rag_eval.json`, `data/evaluation/results/graph_rag_*.json`.
- **Key Reports (`reports/`)**:
  - `reports/graph_rag_evaluation_report.md` (G5 Benchmark Report)
  - `reports/graph_extraction_full_report.md` (G2.2 Full Corpus Extraction Report)
  - `reports/graph_extraction_failed_chunks.json` (76 failed chunks audit)
- **Deployment & Configuration**:
  - `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements.txt`, `.env.example`, `README.md`, `COMMANDS.md`.

---

## 4. Secret & Security Findings

1. **API Keys in Root**:
   - `api_key.text` located at `D:\Programing\RAG\api_key.text` contains 4 raw Google Gemini API keys used across extraction rotation pools.
   - **Remediation**: Exclude from `.gitignore` and `.dockerignore`.
2. **Environment File**:
   - `.env` at `D:\Programing\RAG\.env` contains database passwords and the active `GEMINI_API_KEY`.
   - **Remediation**: Ensure `.env` is ignored. Audit `.env.example` to verify no real keys are included.
3. **Hardcoded Secrets Check in Source Code**:
   - Grep and pattern scan of `app/` and `scripts/` confirmed **zero hardcoded API keys or passwords** in Python source code. All credentials use `os.getenv(...)` with fallback defaults to localhost/development configurations.

---

## 5. File Size Audit & Large File Strategy

GitHub enforces a **100 MB hard limit** on individual files and warns on files larger than **25 MB**.

| File Path | Size | Status / Action |
| :--- | :--- | :--- |
| `data/archived_books/processed/embedded_chunks_full_11609.json` | **136.70 MB** | Exceeds GitHub 100 MB limit $\to$ Exclude via `.gitignore`. |
| `data/processed/embedded_chunks.json` | **67.51 MB** | Exceeds GitHub 50 MB warning $\to$ Exclude via `.gitignore`. |
| `data/archived_books/foundation-models-...pdf` | **18.27 MB** | Exclude via `.gitignore`. |
| `data/raw/Artificial-Intelligence-Index-Report-2024-...pdf` | **13.79 MB** | Local document / External download link in README. |
| `data/archived_books/deep-learning-...pdf` | **13.24 MB** | Exclude via `.gitignore`. |
| `data/archived_books/processed/all_chunks_full_11609.json` | **11.65 MB** | Exclude via `.gitignore`. |
| `data/raw/speech-and-language-processing-...pdf` | **11.14 MB** | Local document / External download link in README. |

---

## 6. Audit Conclusion & Next Steps

The repository is healthy, modular, and structurally sound.
Next actions in Phase G6 execution:
1. Harden `.gitignore` and `.dockerignore`.
2. Sanitize and expand `.env.example`.
3. Resolve the 3 test environment/mock discrepancies so 100% of tests pass.
4. Execute regression verifications for baseline RAG, Graph RAG, FastAPI, and Docker.
5. Publish updated, publication-grade `README.md`.
