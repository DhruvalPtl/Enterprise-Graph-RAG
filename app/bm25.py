"""
BM25 Lexical Retrieval Component for Enterprise RAG.

This module implements the Okapi BM25 ranking algorithm for sparse, keyword-based text retrieval.
Features:
- Regex-based lowercased tokenization.
- Lucene-style non-negative Inverse Document Frequency (IDF) to avoid negative weights on common terms:
  IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
- Standard term saturation (k1 = 1.5) and document length normalization (b = 0.75).
- Flexible initialization from in-memory chunks, JSON files, or PostgreSQL database.
- Standardized SearchResult format preserving all chunk metadata and document attribution.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import math
import re
from collections import Counter
import json
import psycopg
from psycopg.rows import dict_row

from app.config import (
    BM25_K1,
    BM25_B,
    BM25_TOP_K,
    ALL_CHUNKS_FILE,
    EMBEDDED_CHUNKS_FILE,
)
from app.db import get_connection
from app.models import AccessContext

DEMO_ACCESS_METADATA: Dict[str, Dict[str, str]] = {
    "support_faq.txt": {"department": "public", "access_level": "public", "status": "active"},
    "enterprise_platform_architecture.pdf": {"department": "engineering", "access_level": "employee", "status": "active"},
    "ai_governance_policy.md": {"department": "engineering", "access_level": "manager", "status": "active"},
}


def tokenize(text: str) -> List[str]:
    """
    Extracts lowercased alphanumeric word tokens from text.
    Handles None and empty strings safely.
    """
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


def _get_chunk_access(chunk: Dict[str, Any]) -> tuple:
    """Helper to extract (department, access_level, status) from chunk or fallback to DEMO_ACCESS_METADATA."""
    meta = chunk.get("metadata", {})
    doc_name = (
        meta.get("document_name")
        or chunk.get("document_name")
        or ""
    )
    demo = DEMO_ACCESS_METADATA.get(Path(doc_name).name, {})
    dept = meta.get("department") or chunk.get("department") or demo.get("department", "public")
    acc = meta.get("access_level") or chunk.get("access_level") or demo.get("access_level", "public")
    status = meta.get("status") or chunk.get("status") or demo.get("status", "active")
    return dept, acc, status


class BM25Retriever:
    """
    Okapi BM25 Lexical Retriever.
    Indexes a corpus of chunks and scores queries using term frequencies and IDF.
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.doc_freqs: List[Counter] = []
        self.avgdl: float = 0.0
        self.corpus_size: int = 0
        self.idf: Dict[str, float] = {}

        if chunks:
            self.index(chunks)

    def index(self, chunks: List[Dict[str, Any]]) -> "BM25Retriever":
        """
        Indexes a list of chunk dictionaries.
        Computes document lengths, term frequencies, and non-negative IDF table.
        """
        if not chunks:
            self.chunks = []
            self.doc_len = []
            self.doc_freqs = []
            self.avgdl = 0.0
            self.corpus_size = 0
            self.idf = {}
            return self

        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_len = []
        self.doc_freqs = []

        # Count document frequency: in how many documents does term t appear?
        df_counter: Counter = Counter()

        total_length = 0
        for chunk in chunks:
            text = chunk.get("content") or chunk.get("text") or ""
            tokens = tokenize(text)
            length = len(tokens)
            self.doc_len.append(length)
            total_length += length

            tf = Counter(tokens)
            self.doc_freqs.append(tf)

            # Each unique term in this document increments df by 1
            for term in tf:
                df_counter[term] += 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute Lucene-style non-negative Okapi IDF for each term in vocabulary:
        # IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        self.idf = {}
        for term, n_q in df_counter.items():
            numerator = self.corpus_size - n_q + 0.5
            denominator = n_q + 0.5
            self.idf[term] = math.log(1.0 + (numerator / denominator))

        return self

    def search(
        self,
        query_text: str,
        top_k: int = BM25_TOP_K,
        access_context: Optional[AccessContext] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches the indexed corpus for the given query text using BM25,
        enforcing pre-retrieval access control filtering.

        Args:
            query_text: User search phrase or question.
            top_k: Maximum number of ranked results to return.
            access_context: Optional caller access context. Defaults strictly to public access.

        Returns:
            List of ranked chunk dictionaries sorted by descending BM25 score.
            Unauthorized chunks are completely excluded prior to candidate scoring.
        """
        if not query_text or not query_text.strip() or top_k <= 0 or not self.chunks:
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        # Filter query tokens to terms known in vocabulary
        active_tokens = [t for t in query_tokens if t in self.idf]
        if not active_tokens:
            return []

        # Pre-retrieval access control filtering
        ctx = access_context or AccessContext()
        authorized_indices = [
            idx for idx, chunk in enumerate(self.chunks)
            if ctx.is_authorized(*_get_chunk_access(chunk))
        ]
        if not authorized_indices:
            return []

        scores: Dict[int, float] = {idx: 0.0 for idx in authorized_indices}

        for term in active_tokens:
            term_idf = self.idf[term]
            for doc_idx in authorized_indices:
                tf = self.doc_freqs[doc_idx].get(term, 0)
                if tf == 0:
                    continue

                doc_l = self.doc_len[doc_idx]
                norm = 1.0 - self.b + self.b * (doc_l / self.avgdl if self.avgdl > 0 else 1.0)
                term_score = term_idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * norm))
                scores[doc_idx] += term_score

        # Pair scores with chunk indexes and sort descending
        scored_docs = [(score, idx) for idx, score in scores.items() if score > 0.0]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(scored_docs[:top_k], start=1):
            chunk = self.chunks[idx]
            meta = chunk.get("metadata", {})
            doc_name = (
                meta.get("document_name")
                or chunk.get("document_name")
                or f"{chunk.get('document_id', 'doc')}.txt"
            )
            doc_type = (
                meta.get("file_type")
                or meta.get("document_type")
                or Path(doc_name).suffix.lstrip(".")
            )
            dept, acc, stat = _get_chunk_access(chunk)

            results.append({
                "chunk_id": chunk.get("chunk_id", idx),
                "document_id": chunk.get("document_id", "unknown"),
                "content": chunk.get("content") or chunk.get("text", ""),
                "score": round(float(score), 4),
                "rank": rank,
                "source": "bm25",
                "metadata": {
                    "document_name": doc_name,
                    "document_type": doc_type,
                    "page_number": meta.get("page_number", chunk.get("page_number", 1)),
                    "section": meta.get("section", chunk.get("section", "General")),
                    "chunk_index": meta.get("chunk_index", chunk.get("chunk_index", 0)),
                    "department": dept,
                    "access_level": acc,
                    "status": stat,
                },
            })

        return results

    @classmethod
    def from_chunks(
        cls,
        chunks: List[Dict[str, Any]],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "BM25Retriever":
        """Factory creating a BM25Retriever directly from an in-memory chunk list."""
        return cls(chunks=chunks, k1=k1, b=b)

    @classmethod
    def from_file(
        cls,
        file_path: Optional[Path] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "BM25Retriever":
        """
        Factory creating a BM25Retriever by reading from embedded_chunks.json
        (or all_chunks.json).
        """
        if file_path is None:
            file_path = EMBEDDED_CHUNKS_FILE if EMBEDDED_CHUNKS_FILE.exists() else ALL_CHUNKS_FILE
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        return cls(chunks=chunks, k1=k1, b=b)

    @classmethod
    def from_db(
        cls,
        conn: Optional[psycopg.Connection] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "BM25Retriever":
        """
        Factory creating a BM25Retriever by querying active chunks from PostgreSQL.
        Maintains complete alignment between relational, vector, and lexical indices.
        """
        should_close = False
        if conn is None:
            conn = get_connection(autocommit=True)
            should_close = True

        sql = """
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.content,
            c.page_number,
            c.section,
            c.chunk_index,
            d.filename AS document_name,
            d.document_type,
            d.department,
            d.access_level,
            d.status
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY c.id ASC;
        """

        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql)
                rows = cur.fetchall()

            chunks: List[Dict[str, Any]] = [
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "metadata": {
                        "document_name": row["document_name"],
                        "document_type": row["document_type"],
                        "page_number": row["page_number"],
                        "section": row["section"],
                        "chunk_index": row["chunk_index"],
                        "department": row.get("department", "public"),
                        "access_level": row.get("access_level", "public"),
                        "status": row.get("status", "active"),
                    },
                }
                for row in rows
            ]
            return cls(chunks=chunks, k1=k1, b=b)
        finally:
            if should_close:
                conn.close()


def search_bm25(
    query_text: str,
    top_k: int = BM25_TOP_K,
    access_context: Optional[AccessContext] = None,
    retriever: Optional[BM25Retriever] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to perform BM25 search using an existing or auto-loaded retriever.
    """
    if retriever is None:
        try:
            retriever = BM25Retriever.from_db()
        except Exception:
            retriever = BM25Retriever.from_file()

    return retriever.search(query_text=query_text, top_k=top_k, access_context=access_context)
