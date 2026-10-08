"""
Unit and Integration Tests for Gemini LLM Provider.
"""
import os
from unittest.mock import MagicMock
import pytest

from app.llm import GeminiLLMProvider, sanitize_error_message


def test_sanitize_error_message():
    """Verify API keys are redacted from error messages."""
    fake_key = "AIza" + "Sy" + "MockKeyForTesting1234567890abcdef"
    raw_error = f"API call failed with key {fake_key} at endpoint"
    sanitized = sanitize_error_message(raw_error)
    assert fake_key not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized



def test_missing_api_key_raises_error(monkeypatch):
    """Verify clear ValueError when no API key is provided or configured in environment."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("app.llm.GEMINI_API_KEY", None)

    with pytest.raises(ValueError, match="Gemini API key is required"):
        GeminiLLMProvider(api_key=None)

    with pytest.raises(ValueError, match="Gemini API key is required"):
        GeminiLLMProvider(api_key="")


def test_llm_generation_with_mock_client():
    """Verify generate passes prompt and system instruction correctly to client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  Chunking decomposes large documents into coherent passages. [SOURCE 1]  "
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiLLMProvider(client=mock_client, model_name="gemini-2.5-flash")

    prompt = "USER QUESTION:\nWhat is chunking?"
    system_inst = "Answer using only context."

    answer = provider.generate(
        prompt=prompt,
        system_instruction=system_inst,
        temperature=0.2,
        max_output_tokens=512,
    )

    assert answer == "Chunking decomposes large documents into coherent passages. [SOURCE 1]"

    # Verify call parameters
    mock_client.models.generate_content.assert_called_once()
    args, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == prompt
    config = kwargs["config"]
    assert config.system_instruction == system_inst
    assert config.temperature == 0.2
    assert config.max_output_tokens == 512


def test_empty_prompt_raises_value_error():
    """Verify empty or whitespace prompt raises ValueError."""
    mock_client = MagicMock()
    provider = GeminiLLMProvider(client=mock_client)

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        provider.generate(prompt="")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        provider.generate(prompt="   ")


def test_empty_response_raises_runtime_error():
    """Verify empty or None text from API raises RuntimeError."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiLLMProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="empty or invalid generation response"):
        provider.generate(prompt="Hello")


def test_api_failure_raises_runtime_error_safely():
    """Verify API failure is wrapped in RuntimeError and sanitized."""
    mock_client = MagicMock()
    fake_key = "AIza" + "Sy" + "DUMMYKEY1234567890abcdefghijkl"
    mock_client.models.generate_content.side_effect = RuntimeError(
        f"API error 403: Forbidden with key {fake_key}"
    )

    provider = GeminiLLMProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="Gemini generation failed"):
        provider.generate(prompt="test prompt")



def test_repr_hides_api_key():
    """Verify string representation never reveals the API key."""
    provider = GeminiLLMProvider(client=MagicMock(), api_key="secret_test_key_123")
    repr_str = repr(provider)
    assert "secret_test_key_123" not in repr_str
    assert "key_configured=True" in repr_str


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="Live Gemini generation requires GEMINI_API_KEY or GOOGLE_API_KEY",
)
def test_live_gemini_generation():
    """Live integration test: Verifies actual Gemini LLM generation if API key is present."""
    provider = GeminiLLMProvider()
    response = provider.generate(
        prompt="Context: Enterprise Knowledge Intelligence Platform.\nQuestion: What platform is this?",
        system_instruction="You are a helpful assistant. Answer based on the provided context.",
    )
    assert "Enterprise" in response
