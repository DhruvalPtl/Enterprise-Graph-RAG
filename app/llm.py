"""
Google Gemini LLM Generation Provider for Enterprise RAG (Step 6).

Implements grounded generation using the official Google GenAI Python SDK (google-genai).
Ensures safe handling of API credentials, structured system instructions,
temperature control, and clear error diagnostics.
"""
import os
import re
from typing import Optional, Any
from app.config import GEMINI_API_KEY, GEMINI_LLM_MODEL
from app.prompts import SYSTEM_INSTRUCTION


def sanitize_error_message(msg: str) -> str:
    """Removes any accidental API keys from exception strings."""
    return re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", msg)


class GeminiLLMProvider:
    """
    LLM generation service interfacing with Google Gemini models.
    Supports dependency injection of a mock client for testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """
        Initialize GeminiLLMProvider.

        Args:
            api_key: Optional Gemini API key. If not provided, reads GEMINI_API_KEY or GOOGLE_API_KEY.
            model_name: Gemini model name (default: GEMINI_LLM_MODEL, e.g. 'gemini-2.5-flash').
            client: Optional pre-configured genai.Client instance or mock object.
        """
        self.model_name = model_name or GEMINI_LLM_MODEL
        self._api_key = api_key if api_key is not None else (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
        )

        if client is not None:
            self.client = client
        else:
            if not self._api_key or not self._api_key.strip():
                raise ValueError(
                    "Gemini API key is required for LLM generation. "
                    "Please set the GEMINI_API_KEY environment variable or pass api_key."
                )

            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' SDK is required for Gemini generation. "
                    "Please install it via: pip install google-genai"
                )

            self.client = genai.Client(api_key=self._api_key.strip())

    def __repr__(self) -> str:
        """Safe representation hiding credentials."""
        return f"<GeminiLLMProvider model='{self.model_name}' key_configured={bool(self._api_key)}>"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> str:
        """
        Generates text using the Gemini model.

        Args:
            prompt: User/RAG prompt containing retrieved context and question.
            system_instruction: Optional system-level instructions (defaults to SYSTEM_INSTRUCTION).
            temperature: Sampling temperature (default: 0.0 for deterministic, factual output).
            max_output_tokens: Maximum tokens in response.

        Returns:
            Generated response text.

        Raises:
            ValueError: If prompt is empty.
            RuntimeError: If generation fails or returns empty content.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        sys_inst = system_instruction if system_instruction is not None else SYSTEM_INSTRUCTION

        candidate_models = [self.model_name]
        for alt in ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"]:
            if alt not in candidate_models:
                candidate_models.append(alt)

        import time

        last_error = None
        for model_to_try in candidate_models:
            for attempt in range(3):
                try:
                    from google.genai import types

                    config = types.GenerateContentConfig(
                        system_instruction=sys_inst,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    )

                    response = self.client.models.generate_content(
                        model=model_to_try,
                        contents=prompt,
                        config=config,
                    )

                    if response and getattr(response, "text", None):
                        self.model_name = model_to_try
                        return response.text.strip()

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    # Transient network / DNS glitches (e.g. Errno 11001 getaddrinfo)
                    is_network_error = any(k in err_str for k in ("11001", "getaddrinfo", "ConnectError", "timeout", "timed out"))
                    if is_network_error and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    # If it's a 503 capacity, 429 quota exhaustion, or 404 model not found, try next candidate model
                    is_retriable = any(k in err_str for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "404"))
                    if is_retriable and model_to_try != candidate_models[-1]:
                        break
                    break

        if last_error:
            err_msg = sanitize_error_message(str(last_error))
            raise RuntimeError(f"Gemini generation failed: {err_msg}") from None

        raise RuntimeError("Gemini API returned an empty or invalid generation response.")

