"""
Configuration settings for the Document Ingestion and Chunking pipeline.
"""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    # Load environment variables from .env if present
    load_dotenv(override=False)
except ImportError:
    pass

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", BASE_DIR / "data" / "processed"))

# Chunking Configuration
# Default target chunk size (in characters)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 800))

# Overlap size between adjacent chunks (in characters) to retain context across cuts
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 150))

# Supported File Extensions
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

# Embedding Configuration (Step 2 & Step 2A)
# Active provider: 'sentence-transformers' (PRIMARY/default) or 'gemini' (secondary/alternative)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip().lower()

# Local primary configuration
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Gemini secondary configuration
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY == "your_gemini_api_key_here":
    GEMINI_API_KEY = None

DEFAULT_EMBEDDING_MODEL = (
    LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER in ("sentence-transformers", "local", "minilm")
    else GEMINI_EMBEDDING_MODEL
)

ALL_CHUNKS_FILE = PROCESSED_DATA_DIR / "all_chunks.json"
EMBEDDED_CHUNKS_FILE = PROCESSED_DATA_DIR / "embedded_chunks.json"

# PostgreSQL Database Configuration (Step 3 & Step 8)
POSTGRES_HOST = os.getenv("DATABASE_HOST") or os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("DATABASE_PORT") or os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("DATABASE_NAME") or os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("DATABASE_USER") or os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("DATABASE_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres")
if POSTGRES_PASSWORD == "your_postgres_password_here":
    POSTGRES_PASSWORD = "postgres"
DATABASE_URL = os.getenv("DATABASE_URL")

# Aliases for generic database config naming
DATABASE_HOST = POSTGRES_HOST
DATABASE_PORT = POSTGRES_PORT
DATABASE_NAME = POSTGRES_DB
DATABASE_USER = POSTGRES_USER
DATABASE_PASSWORD = POSTGRES_PASSWORD

# Vector Search Configuration (Step 3)
VECTOR_DIMENSION = 384

# Hybrid Retrieval & RRF Configuration (Step 4)
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 20))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
RRF_TOP_K = int(os.getenv("RRF_TOP_K", 20))
RRF_K = int(os.getenv("RRF_K", 60))
BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B = float(os.getenv("BM25_B", 0.75))

# Cross-Encoder Reranking Configuration (Step 5)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", 20))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", 5))

# Generation & Context Builder Configuration (Step 6)
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash-lite")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 4000))

# FastAPI Server Configuration (Step 7)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
API_TITLE = "Enterprise Knowledge Intelligence Platform API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-oriented Enterprise RAG API featuring hybrid retrieval, "
    "Reciprocal Rank Fusion, Cross-Encoder reranking, and grounded Gemini LLM generation."
)


