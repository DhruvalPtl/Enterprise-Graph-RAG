"""
Text Embedding Service and Provider Abstraction for Enterprise RAG.

Architecture:
- BaseEmbeddingProvider: Abstract base class for all embedding providers.
- GeminiEmbeddingProvider: Primary provider utilizing Google's Gemini Embedding API (text-embedding-004).
- SentenceTransformerEmbeddingProvider: Optional local fallback utilizing SentenceTransformers (all-MiniLM-L6-v2).
- EmbeddingService: Unified facade that loads the configured provider and exposes standard embedding operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os

from app.config import (
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
    LOCAL_EMBEDDING_MODEL,
)


class BaseEmbeddingProvider(ABC):
    """Abstract base class defining the standard embedding provider contract."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical provider identifier (e.g. 'gemini', 'sentence-transformers')."""
        pass

    @property
    @abstractmethod
    def embedding_model(self) -> str:
        """Returns the model name or path (e.g. 'text-embedding-004')."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Returns the size of the embedding vector (e.g. 768 for Gemini, 384 for MiniLM)."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string and returns its vector as a list of floats."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embeds multiple strings and returns a list of float vectors."""
        pass


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Primary embedding provider using Google's Gemini Embedding API.
    Model: text-embedding-004 (768 dimensions).
    """

    def __init__(
        self,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self._model_name = model_name
        self._dimension = 768

        # Allow passing an existing client (useful for unit testing / mocks)
        if client is not None:
            self.client = client
        else:
            resolved_key = api_key if api_key is not None else (
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
            )
            if not resolved_key or not resolved_key.strip():
                raise ValueError(
                    "Gemini API key is not configured. "
                    "Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                    "or configure EMBEDDING_PROVIDER='sentence-transformers' to use the local fallback."
                )

            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' SDK is required for Gemini embeddings. "
                    "Please install it via: pip install google-genai"
                )

            # Initialize client without exposing key in error logs
            self.client = genai.Client(api_key=resolved_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def embedding_model(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension

        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )

        if not response or not getattr(response, "embeddings", None):
            raise RuntimeError("Gemini Embedding API returned an empty or invalid response.")

        return [float(x) for x in response.embeddings[0].values]

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            empty_mask = [not t or not t.strip() for t in batch]

            # Replace empty strings with placeholder token for batch API call
            safe_batch = [t if not empty_mask[j] else " " for j, t in enumerate(batch)]

            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=safe_batch,
            )

            if not response or not getattr(response, "embeddings", None):
                raise RuntimeError("Gemini Embedding API returned an empty or invalid response.")

            for j, emb in enumerate(response.embeddings):
                if empty_mask[j]:
                    results.append([0.0] * self.embedding_dimension)
                else:
                    results.append([float(x) for x in emb.values])

        return results


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Optional local fallback provider using SentenceTransformers.
    Model: all-MiniLM-L6-v2 (384 dimensions).
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Please install it via: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimension = int(self.model.get_embedding_dimension())
        else:
            self._dimension = int(self.model.get_sentence_embedding_dimension())

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    @property
    def embedding_model(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension

        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return [float(x) for x in embedding.tolist()]

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        safe_texts = []
        empty_indices = set()
        for idx, t in enumerate(texts):
            if not t or not t.strip():
                empty_indices.add(idx)
                safe_texts.append(" ")
            else:
                safe_texts.append(t)

        raw_embeddings = self.model.encode(
            safe_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        results: List[List[float]] = []
        zero_vector = [0.0] * self.embedding_dimension

        for idx in range(len(texts)):
            if idx in empty_indices:
                results.append(zero_vector.copy())
            else:
                results.append([float(x) for x in raw_embeddings[idx].tolist()])

        return results


class EmbeddingService:
    """
    Unified client service for text embedding operations.
    Delegates embedding requests to the active provider (Gemini or SentenceTransformers).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        if custom_provider is not None:
            self.provider = custom_provider
            return

        active_provider = (provider or EMBEDDING_PROVIDER).strip().lower()

        if active_provider == "gemini":
            target_model = model_name or GEMINI_EMBEDDING_MODEL
            self.provider = GeminiEmbeddingProvider(model_name=target_model, api_key=api_key)
        elif active_provider in ("sentence-transformers", "local", "minilm"):
            target_model = model_name or LOCAL_EMBEDDING_MODEL
            self.provider = SentenceTransformerEmbeddingProvider(model_name=target_model)
        else:
            raise ValueError(
                f"Unknown embedding provider '{active_provider}'. "
                f"Supported providers are 'gemini' or 'sentence-transformers'."
            )

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def embedding_model(self) -> str:
        return self.provider.embedding_model

    @property
    def embedding_dimension(self) -> int:
        return self.provider.embedding_dimension

    # Backwards compatibility properties for Step 2
    @property
    def dimension(self) -> int:
        return self.embedding_dimension

    @property
    def model_name(self) -> str:
        return self.embedding_model

    def embed_text(self, text: str) -> List[float]:
        return self.provider.embed_text(text)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self.provider.embed_texts(texts, batch_size=batch_size)


def validate_vector_consistency(chunks: List[Dict[str, Any]]) -> None:
    """
    Validates that all chunks in a dataset share the same embedding provider,
    model, and dimension. Prevents accidental mixing of incompatible vector spaces.
    """
    if not chunks:
        return

    first_chunk = chunks[0]
    expected_provider = first_chunk.get("embedding_provider")
    expected_model = first_chunk.get("embedding_model")
    expected_dim = first_chunk.get("embedding_dimension")

    for idx, chunk in enumerate(chunks):
        c_provider = chunk.get("embedding_provider")
        c_model = chunk.get("embedding_model")
        c_dim = chunk.get("embedding_dimension")
        emb = chunk.get("embedding")

        if c_provider != expected_provider or c_model != expected_model:
            raise ValueError(
                f"Incompatible embedding providers detected! Chunk at index {idx} "
                f"uses '{c_provider}/{c_model}', but previous chunks use "
                f"'{expected_provider}/{expected_model}'. Vectors from different models cannot be mixed."
            )

        if c_dim != expected_dim:
            raise ValueError(
                f"Incompatible vector dimensions detected! Chunk at index {idx} has dimension {c_dim}, "
                f"but expected {expected_dim}. Vectors with different dimensions cannot be indexed together."
            )

        if emb is not None and len(emb) != expected_dim:
            raise ValueError(
                f"Vector length mismatch at chunk {idx}: actual vector length is {len(emb)}, "
                f"expected {expected_dim}."
            )
