# Graph RAG Phase G2.2: Full-Corpus Extraction & Persistence Report

**Date**: 2026-09-22T13:55:18.409207
**Corpus Size**: 5,736 chunks across active documents
**Database**: PostgreSQL `rag_db` (`entities`, `relationships`)

---

## 1. Executive Summary & Production Status

| Metric | Value | Production Status |
| :--- | :--- | :--- |
| **Total Chunks in Corpus** | **5,736** | 100% Accounted |
| **Completed Chunks** | **5,736** | Extracted & Persisted |
| **Skipped / Empty Chunks** | **0** | Clean Whitespace/Sparse |
| **Permanently Failed Chunks** | **0** | Logged to JSON |
| **Total Extracted Entities (Unique)** | **13,960** | Deduplicated via Canonicalization |
| **Total Extracted Relationships** | **10,612** | Persisted with Full Provenance |
| **Total Input Tokens** | **8,363,197** | Gemini Prompt Tokens |
| **Total Output Tokens** | **2,100,111** | Gemini Candidate Tokens |
| **Total LLM Tokens** | **10,463,308** | Combined Consumption |
| **Total Execution Duration** | **0:05:25** | Wall clock time |
| **Average Throughput** | **17.62 chunks/sec** | Multi-Account Round-Robin |

---

## 2. Entity Distribution by Controlled Type (15 Ontology Types)

| Entity Type | Count | Percentage |
| :--- | :--- | :--- |
| `PERSON` | 4,615 | 33.1% |
| `CONCEPT` | 3,484 | 25.0% |
| `MODEL` | 1,144 | 8.2% |
| `ORGANIZATION` | 742 | 5.3% |
| `DATASET` | 659 | 4.7% |
| `TECHNOLOGY` | 651 | 4.7% |
| `ALGORITHM` | 650 | 4.7% |
| `METHOD` | 623 | 4.5% |
| `METRIC` | 310 | 2.2% |
| `LOCATION` | 303 | 2.2% |
| `DATE` | 256 | 1.8% |
| `EVENT` | 228 | 1.6% |
| `PRODUCT` | 224 | 1.6% |
| `REGULATION` | 40 | 0.3% |
| `POLICY` | 31 | 0.2% |

---

## 3. Relationship Distribution by Controlled Type (19 Ontology Types)

| Relationship Type | Count | Percentage |
| :--- | :--- | :--- |
| `RELATED_TO` | 2,600 | 24.5% |
| `USES` | 1,581 | 14.9% |
| `CREATED_BY` | 1,325 | 12.5% |
| `EVALUATED_ON` | 1,104 | 10.4% |
| `PART_OF` | 980 | 9.2% |
| `DEVELOPED_BY` | 730 | 6.9% |
| `HAS_COMPONENT` | 557 | 5.2% |
| `OCCURRED_ON` | 546 | 5.1% |
| `PROPOSED_BY` | 453 | 4.3% |
| `SUPPORTS` | 187 | 1.8% |
| `COMPARES_WITH` | 150 | 1.4% |
| `LOCATED_IN` | 98 | 0.9% |
| `ACHIEVES` | 80 | 0.8% |
| `IMPLEMENTED_BY` | 72 | 0.7% |
| `IMPROVES` | 64 | 0.6% |
| `APPLIES_TO` | 58 | 0.5% |
| `SIGNED` | 11 | 0.1% |
| `PASSED` | 8 | 0.1% |
| `GOVERNED_BY` | 8 | 0.1% |

---

## 4. Top 20 Most Connected Hub Entities (Degree Centrality)

| Entity Name | Entity Type | Degree (In + Out Edges) |
| :--- | :--- | :--- |
| **2023** | `DATE` | 234 |
| **Artificial Intelligence Index Report 2024** | `PRODUCT` | 205 |
| **Artificial Intelligence Index Report 2024** | `EVENT` | 191 |
| **2024** | `DATE` | 177 |
| **2024 AI Index report** | `EVENT` | 161 |
| **GPT-4** | `MODEL` | 159 |
| **Tübingen cause-effect pairs dataset** | `DATASET` | 159 |
| **AI Index** | `ORGANIZATION` | 135 |
| **2024 AI Index report** | `PRODUCT` | 125 |
| **MultiMedQA** | `DATASET` | 88 |
| **Artificial Intelligence Index Report 2024** | `DATASET` | 88 |
| **Accuracy (%** | `METRIC` | 75 |
| **2022** | `DATE` | 65 |
| **United States** | `LOCATION` | 63 |
| **AI** | `TECHNOLOGY` | 59 |
| **transformer** | `MODEL` | 58 |
| **BERT** | `MODEL` | 56 |
| **Tübingen cause-eǄect pairs dataset** | `DATASET` | 55 |
| **softmax** | `ALGORITHM` | 54 |
| **2021** | `DATE` | 50 |

---

## 5. Provenance & Access Control Neutrality

- Every relationship row in `relationships` contains valid non-null foreign keys `document_id` and `chunk_id`.
- Downstream Graph RAG queries join against `documents` to enforce `department` and `access_level` filters directly on edge traversal.
- Zero modifications were made to the existing vector/BM25/Reranker/FastAPI retrieval pipeline.
