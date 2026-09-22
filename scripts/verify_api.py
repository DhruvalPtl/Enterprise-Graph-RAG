"""
Verification script for Step 7 FastAPI REST API.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.api.main import app

def main():
    print("=" * 80)
    print("  FASTAPI REST API VERIFICATION (Step 7)")
    print("=" * 80)

    with TestClient(app) as client:
        # 1. Health Endpoint
        health_res = client.get("/health")
        print(f"\n[1] GET /health -> Status: {health_res.status_code}")
        print(f"    Payload: {health_res.json()}")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "ok"
        assert health_res.json()["database"] == "connected"

        # 2. Documentation Endpoints
        docs_res = client.get("/docs")
        print(f"\n[2] GET /docs -> Status: {docs_res.status_code} ({len(docs_res.text)} bytes)")
        assert docs_res.status_code == 200

        openapi_res = client.get("/openapi.json")
        print(f"    GET /openapi.json -> Status: {openapi_res.status_code}")
        assert openapi_res.status_code == 200
        assert "/query" in openapi_res.json()["paths"]

        # 3. Valid POST /query
        print("\n[3] POST /query -> Question: 'What is chunking and chunk overlap strategy?'")
        q1_res = client.post("/query", json={
            "query": "What is chunking and chunk overlap strategy?",
            "top_k": 3,
        })
        print(f"    Status: {q1_res.status_code}")
        assert q1_res.status_code == 200
        q1_data = q1_res.json()
        print(f"    Model Used : {q1_data.get('model_name')}")
        print(f"    Answer     :\n{q1_data['answer']}")
        print("    Citations  :")
        for c in q1_data["citations"]:
            score_str = f"(Reranker: {c['reranker_score']:.4f})" if c.get("reranker_score") is not None else ""
            print(f"      - {c['formatted']} {score_str}")

        # 4. Out-of-Corpus Query (No-Evidence Refusal)
        print("\n[4] POST /query -> Question: 'What is the capital of Mars?'")
        q2_res = client.post("/query", json={
            "query": "What is the capital of Mars?",
            "top_k": 3,
        })
        print(f"    Status: {q2_res.status_code}")
        assert q2_res.status_code == 200
        q2_data = q2_res.json()
        print(f"    Answer : {q2_data['answer']}")

    print("\n" + "=" * 80)
    print("FastAPI API verification completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
