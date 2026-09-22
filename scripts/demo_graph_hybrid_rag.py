"""
Demonstration Script: Graph + Vector Hybrid RAG (Phase G4).

Demonstrates how Graph Retrieval and Vector/BM25 Retrieval complement each other:
- Query: 'What organization developed Gemini Ultra and what benchmark was it evaluated on?'
- Displays:
  1. Vector & BM25 candidate chunks
  2. Graph extracted entities, relationships, and provenance chunks
  3. Fused candidates showing deduplication and provenance attribution ['vector', 'graph']
  4. Final Cross-Encoder reranked evidence passages
  5. Authoritative grounded citations
  6. Grounded response from the RAG Pipeline
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db import get_connection
from app.models import AccessContext
from app.hybrid_retriever import GraphVectorHybridRetriever
from app.rag import RAGPipeline


def run_hybrid_rag_demo():
    query = "What organization developed Gemini Ultra and what benchmark was it evaluated on?"
    print("\n" + "=" * 80)
    print("DEMO: GRAPH + VECTOR HYBRID RAG (Phase G4)")
    print("=" * 80)
    print(f"QUERY: \"{query}\"\n")

    conn = get_connection()
    try:
        retriever = GraphVectorHybridRetriever()
        result = retriever.retrieve_fused(
            query=query,
            candidate_k=10,
            top_k=3,
            access_context=AccessContext(),
            conn=conn,
        )

        print("-" * 80)
        print("1. VECTOR / BM25 RETRIEVAL BRANCH")
        print("-" * 80)
        print(f"Total vector candidates retrieved: {len(result.vector_candidates)}")
        for i, c in enumerate(result.vector_candidates[:3], start=1):
            print(f"  [{i}] Chunk ID: {c.chunk_id} | Page: {c.metadata.get('page_number')} | RRF Score: {c.rrf_score}")
            print(f"      Snippet: {repr(c.content[:90])}...")

        print("\n" + "-" * 80)
        print("2. GRAPH RETRIEVAL BRANCH (Phase G3)")
        print("-" * 80)
        if result.graph_result:
            print(f"Matched Seed Entities: {[e.canonical_name for e in result.graph_result.matched_entities]}")
            print(f"Graph Relationships Found: {len(result.graph_result.relationships)}")
            for rel in result.graph_result.relationships[:5]:
                src = rel.metadata.get("source_name")
                tgt = rel.metadata.get("target_name")
                print(f"  - ({src}) --[{rel.relationship_type}]--> ({tgt})")
                print(f"    Provenance: Chunk {rel.chunk_id} (Page {rel.page_number})")
            print(f"Connected Entities Discovered: {[e.canonical_name for e in result.graph_result.connected_entities[:5]]}")
            print(f"Graph Provenance Chunks: {result.graph_result.source_chunk_ids}")

        print("\n" + "-" * 80)
        print("3. EVIDENCE FUSION & DEDUPLICATION (Phase G4)")
        print("-" * 80)
        print(f"Deduplication Statistics: {result.deduplication_stats}")
        print(f"Total Fused Candidates: {len(result.fused_candidates)}")
        overlap_chunks = [c for c in result.fused_candidates if "graph" in c.sources.get("retrieved_by", []) and "vector" in c.sources.get("retrieved_by", [])]
        print(f"Chunks retrieved by BOTH Vector AND Graph: {[c.chunk_id for c in overlap_chunks]}")
        for oc in overlap_chunks[:2]:
            print(f"  * Chunk {oc.chunk_id}: RetrievedBy={oc.sources.get('retrieved_by')}")
            if oc.metadata.get("graph_relationships"):
                for gr in oc.metadata["graph_relationships"][:2]:
                    print(f"    Rel: {gr['source']} -> {gr['relationship_type']} -> {gr['target']}")

        print("\n" + "-" * 80)
        print("4. CROSS-ENCODER RERANKED EVIDENCE (Top-K)")
        print("-" * 80)
        for r in result.reranked_results:
            print(f"  Rank #{r.rank} | Chunk {r.chunk_id} | Score: {r.reranker_score:.4f} | Source: {r.source} | RetrievedBy: {r.sources.get('retrieved_by')}")
            print(f"  Page: {r.metadata.get('page_number')} | File: {r.metadata.get('document_name')}")
            if r.metadata.get("graph_relationships"):
                for gr in r.metadata["graph_relationships"]:
                    print(f"  [GRAPH] {gr['source']} --[{gr['relationship_type']}]--> {gr['target']}")
            print(f"  Content: {repr(r.content[:130])}...\n")

        print("-" * 80)
        print("5. END-TO-END RAG PIPELINE EXECUTION")
        print("-" * 80)
        pipeline = RAGPipeline(retrieval_pipeline=retriever)
        response = pipeline.answer_query(query=query, top_k=3, conn=conn)

        print(f"ANSWER:\n{response.answer}\n")
        print("AUTHORITATIVE CITATIONS:")
        for c in response.citations:
            print(f"  [{c.source_id}] {c.filename} | Page {c.page_number} | Chunk {c.chunk_id}")
            print(f"      Retrieved via: {c.sources.get('retrieved_by', ['unknown'])}")

        print("\nDIAGNOSTICS:")
        for k, v in response.diagnostics.items():
            print(f"  {k}: {v}")
        print("=" * 80 + "\n")

    finally:
        conn.close()


if __name__ == "__main__":
    run_hybrid_rag_demo()
