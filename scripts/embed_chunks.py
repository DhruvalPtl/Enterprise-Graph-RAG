"""
Generates embeddings for processed text chunks (Step 2A).

Workflow:
1. Loads raw chunks from data/processed/all_chunks.json.
2. Extracts chunk text.
3. Generates dense embeddings using the configured provider:
   - Primary: Google Gemini API (text-embedding-004, 768 dimensions)
   - Fallback: SentenceTransformers (all-MiniLM-L6-v2, 384 dimensions)
4. Preserves all original chunk text, document IDs, and metadata.
5. Records embedding_provider, embedding_model, and embedding_dimension.
6. Validates vector consistency to prevent mixing vector spaces.
7. Saves the enriched chunks to data/processed/embedded_chunks.json.

Usage:
    python scripts/embed_chunks.py
    python scripts/embed_chunks.py --provider sentence-transformers
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so app modules import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    ALL_CHUNKS_FILE,
    EMBEDDED_CHUNKS_FILE,
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    GEMINI_API_KEY,
)
from app.embeddings import EmbeddingService, validate_vector_consistency


def embed_chunks(
    input_file: Path = ALL_CHUNKS_FILE,
    output_file: Path = EMBEDDED_CHUNKS_FILE,
    provider: str = EMBEDDING_PROVIDER,
    model_name: str = None,
):
    active_provider = provider.strip().lower()
    active_model = model_name or (
        GEMINI_EMBEDDING_MODEL if active_provider == "gemini" else LOCAL_EMBEDDING_MODEL
    )

    print("=" * 70)
    print("  Enterprise RAG - Text Embedding Generation (Step 2A)")
    print("=" * 70)
    print(f"  Active Provider    : {active_provider}")
    print(f"  Embedding Model    : {active_model}")
    print(f"  Input Chunks File  : {input_file}")
    print(f"  Output File        : {output_file}")
    print("-" * 70)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input chunks file not found: {input_file}. "
            f"Please run 'python run_pipeline.py' first to generate chunks."
        )

    # 1. Verify credentials if using Gemini
    if active_provider == "gemini" and (not GEMINI_API_KEY or not GEMINI_API_KEY.strip()):
        print("\n[CONFIGURATION ERROR]")
        print("  GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        print("  To use Gemini embeddings:")
        print("    1. Set GEMINI_API_KEY in your environment or in a .env file.")
        print("    2. Or run with local fallback: python scripts/embed_chunks.py --provider sentence-transformers")
        print("=" * 70)
        sys.exit(1)

    # 2. Read existing chunks
    with open(input_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    total_chunks = len(chunks_data)
    print(f"Loaded {total_chunks} chunks from {input_file.name}")

    if total_chunks == 0:
        print("Warning: No chunks found to embed.")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        return

    # 3. Initialize embedding service
    print(f"Initializing embedding provider '{active_provider}' with model '{active_model}'...")
    service = EmbeddingService(provider=active_provider, model_name=active_model)
    dimension = service.embedding_dimension
    print(f"Provider ready. Embedding dimension: {dimension}")

    # 4. Extract text from each chunk
    texts = [chunk.get("text", "") for chunk in chunks_data]

    # 5. Generate embeddings in batch
    print(f"Generating embeddings for {total_chunks} chunks...")
    embeddings = service.embed_texts(texts)

    # 6. Attach embedding and provider provenance to each chunk
    embedded_chunks = []
    for chunk, emb in zip(chunks_data, embeddings):
        enriched_chunk = dict(chunk)
        enriched_chunk["embedding"] = emb
        enriched_chunk["embedding_provider"] = service.provider_name
        enriched_chunk["embedding_model"] = service.embedding_model
        enriched_chunk["embedding_dimension"] = dimension
        embedded_chunks.append(enriched_chunk)

    # 7. Validate vector consistency before saving
    validate_vector_consistency(embedded_chunks)

    # 8. Save to output JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print("\n[EMBEDDING SUMMARY]")
    print(f"  Embedding Provider    : {service.provider_name}")
    print(f"  Model Used            : {service.embedding_model}")
    print(f"  Embedding Dimension   : {dimension}")
    print(f"  Chunks Embedded       : {len(embedded_chunks)}")
    print(f"  Output Saved To       : {output_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate dense text embeddings for document chunks.")
    parser.add_argument(
        "--provider",
        type=str,
        default=EMBEDDING_PROVIDER,
        help="Embedding provider ('gemini' or 'sentence-transformers')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name (defaults to provider's configured model)",
    )
    args = parser.parse_args()

    embed_chunks(provider=args.provider, model_name=args.model)


if __name__ == "__main__":
    main()
