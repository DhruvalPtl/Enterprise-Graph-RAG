# Enterprise Graph RAG - User Interface

A modular, professional Streamlit web interface for the **Enterprise Graph RAG** platform.

The UI communicates directly with the FastAPI REST backend (`POST /query` and `GET /health`), providing complete transparency into the retrieval lifecycle, knowledge graph traversals, and source citations.

---

## Key Capabilities

1. **Conversational Chatbot Assistant**
   - Streamlit chat interface (`st.chat_message`, `st.chat_input`) with conversation history.
   - Quick-start example prompt chips to immediately test domain-specific questions.
   - Grounded answers with highlighted `[SOURCE X]` citation tags.
   - Accordions for authoritative source citations, graph traversal paths, and pipeline execution telemetry.

2. **Knowledge Base File Explorer**
   - Inspect active documents loaded in PostgreSQL with chunk counts, file format, and RBAC tags.
   - Read original file content: page-by-page PDF viewer with original download button, rendered Markdown, and text viewer.
   - Inspect individual indexed chunks and section headings.
   - Delete documents with cascading cleanup across chunks and graph relationships.

3. **Document Ingestion Hub**
   - Upload and process `.pdf`, `.md`, and `.txt` files directly in the browser.
   - Assign department (`public`, `engineering`, `finance`, `hr`) and clearance level (`public`, `employee`, `manager`, `admin`).
   - Automatically computes 384-dimensional dense vectors using MiniLM and writes to PostgreSQL pgvector with HNSW cosine index.
   - Optional Knowledge Graph extraction via Gemini.

4. **Dual-Engine Retrieval & RBAC Control**
   - Switch between **Hybrid Mode** (Dense Vector + BM25 + PostgreSQL Knowledge Graph) and **Baseline Mode** (Dense + BM25 only).
   - Configure caller organizational department and clearance level to test access control enforcement.

---

## Directory Structure

```
ui/
├── __init__.py
├── app.py                     # Main Streamlit application with sidebar navigation
├── api_client.py              # HTTP client (httpx) for FastAPI /query & /health
├── ingestion_helper.py        # Document parsing, embedding, and database persistence
├── views/
│   ├── __init__.py
│   ├── chat_view.py           # Conversational assistant & chat message flow
│   ├── files_view.py          # Knowledge base explorer & original document viewer
│   └── ingest_view.py         # Full-page document uploader & indexer
├── components/
│   ├── __init__.py
│   ├── header.py              # Application branding and live status indicator
│   ├── answer_view.py         # Grounded answer and zero-hallucination refusal banner
│   ├── citations_view.py      # Source cards with provenance badges & passage inspector
│   ├── graph_evidence_view.py # Knowledge graph triple visualization
│   └── diagnostics_view.py    # Latency breakdown, candidate counts, and raw telemetry
└── styles/
    └── main.css               # Design tokens, high-contrast colors, and badge styling
```

---

## Quickstart

### 1. Start the FastAPI Backend
```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
Verify the backend is live at `http://localhost:8000/docs`.

### 2. Start the Streamlit UI
```powershell
streamlit run ui/app.py --server.port 8501
```

The application opens at `http://localhost:8501`.
