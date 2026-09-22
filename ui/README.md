# Enterprise Graph RAG - User Interface (Phase G7)

A modular, professional Streamlit web interface for the **Enterprise Graph RAG** platform.

The UI communicates directly with the FastAPI REST backend (`POST /query` and `GET /health`), providing complete transparency into the retrieval lifecycle, knowledge graph traversals, and source citations.

---

## Key Capabilities

1. **Dual-Engine Retrieval Control**
   - Switch seamlessly between **Hybrid Mode** (Dense Vector + BM25 + PostgreSQL Knowledge Graph) and **Baseline Mode** (Dense + BM25 only).
   - Adjust final citation limits (`top_k`) and Stage 1 retrieval candidate pool size (`candidate_k`).

2. **Source Provenance Badges**
   - Visual attribution badges for every evidence citation:
     - <span style="color:#38bdf8; font-weight:bold;">Dense / BM25</span>: Retrieved via semantic embedding or keyword matching.
     - <span style="color:#c084fc; font-weight:bold;">Graph Traversal</span>: Retrieved via PostgreSQL entity/relationship traversal.
     - <span style="color:#34d399; font-weight:bold;">Hybrid Verified</span>: Retrieved independently by **both** vector and graph pipelines.

3. **Knowledge Graph Traversal Inspection**
   - Displays query seed entities recognized in natural language.
   - Shows traversed relationship triples `(Source) ──[RELATION]──► (Target)` with database chunk, document, and page provenance.

4. **Passage Grounding & Refusal Guardrails**
   - Displays grounded Gemini answers with highlighted `[SOURCE X]` citation tags.
   - Expandable passage inspection to verify the raw text used by the model.
   - Refusal alert banner when zero grounded evidence exists, preventing hallucinations.

5. **Operational Telemetry & Latency Breakdown**
   - Real-time latency tracking across vector search, graph search, cross-encoder reranking, and LLM generation.
   - Candidate pool deduplication counts and token usage breakdown.

6. **Role-Based Access Control (RBAC)**
   - Configurable caller department and clearance level (`public`, `employee`, `manager`, `admin`).

---

## Directory Structure

```
ui/
├── __init__.py
├── app.py                     # Main Streamlit application entry point
├── api_client.py              # HTTP client (httpx) for FastAPI /query & /health
├── styles/
│   └── main.css               # Design tokens, badges, and card styling
├── components/
│   ├── __init__.py
│   ├── header.py              # Application branding and live status indicator
│   ├── query_input.py         # Search box & real corpus preset chips
│   ├── answer_view.py         # Grounded answer and zero-hallucination refusal banner
│   ├── citations_view.py      # Source cards with provenance badges & passage inspector
│   ├── graph_evidence_view.py # Knowledge graph triple visualization
│   └── diagnostics_view.py    # Latency breakdown, candidate counts, and raw telemetry
└── README.md                  # This documentation
```

---

## Quickstart

### Prerequisites
Ensure dependencies are installed:
```powershell
pip install -r requirements.txt
```

### 1. Start the FastAPI Backend
In your first terminal, launch the backend API:
```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
Verify the backend is live at `http://localhost:8000/docs`.

### 2. Start the Streamlit UI
In your second terminal, launch the Streamlit frontend:
```powershell
streamlit run ui/app.py --server.port 8501
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Configuration

| Environment Variable | Default Value | Description |
|---|---|---|
| `RAG_API_URL` | `http://localhost:8000` | Base URL of the running FastAPI service |

You can also change the API URL directly in the UI sidebar without restarting the server.
