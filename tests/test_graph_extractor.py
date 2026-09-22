"""
Unit tests for Graph RAG Phase G2.1 LLM Extraction Service with Mock Gemini Client.
Zero live API / network dependencies.
"""
import json
import pytest
from unittest.mock import MagicMock

from app.graph_extractor import (
    GraphExtractorService,
    parse_raw_extraction_json,
    EXTRACTION_SYSTEM_PROMPT,
)
from app.graph_ontology import RawExtractionOutput, ValidatedExtractionResult


class TestRawExtractionJsonParsing:
    """Tests parsing and defensive handling of raw LLM JSON outputs."""

    def test_parse_valid_json(self):
        sample = {
            "entities": [
                {"name": "BERT", "entity_type": "MODEL", "confidence": 1.0}
            ],
            "relationships": []
        }
        res = parse_raw_extraction_json(json.dumps(sample))
        assert len(res.entities) == 1
        assert res.entities[0].name == "BERT"
        assert res.entities[0].entity_type == "MODEL"

    def test_parse_json_in_markdown_fences(self):
        fenced = """```json
{
  "entities": [
    {"name": "wav2vec 2.0", "entity_type": "MODEL"}
  ],
  "relationships": []
}
```"""
        res = parse_raw_extraction_json(fenced)
        assert len(res.entities) == 1
        assert res.entities[0].name == "wav2vec 2.0"

    def test_parse_empty_string(self):
        res = parse_raw_extraction_json("")
        assert res.entities == []
        assert res.relationships == []

    def test_parse_malformed_json_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            parse_raw_extraction_json("Not valid json {{{")
        assert "Failed to parse LLM response as JSON" in str(excinfo.value)

    def test_parse_non_dict_json_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            parse_raw_extraction_json('["item1", "item2"]')
        assert "must be a JSON object" in str(excinfo.value)


class TestGraphExtractorService:
    """Tests the extraction service with mocked Gemini clients."""

    def test_successful_extraction_flow(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "entities": [
                {"name": "U.S. Senate", "entity_type": "ORGANIZATION"},
                {"name": "Outbound Investment Transparency Act", "entity_type": "REGULATION"},
            ],
            "relationships": [
                {
                    "source": "U.S. Senate",
                    "target": "Outbound Investment Transparency Act",
                    "relationship_type": "PASSED",
                    "evidence": "U.S. Senate passes Outbound Investment Transparency Act"
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        service = GraphExtractorService(client=mock_client, model_name="mock-model")

        chunk_text = "On July 25, 2023, the U.S. Senate passes the Outbound Investment Transparency Act."
        result = service.extract_from_chunk(
            chunk_id=8273,
            document_id=91,
            page_number=372,
            chunk_text=chunk_text,
        )

        assert isinstance(result, ValidatedExtractionResult)
        assert result.status == "success"
        assert result.document_id == 91
        assert result.chunk_id == 8273
        assert result.page_number == 372

        # Check entity normalization and canonicalization
        assert len(result.entities) == 2
        names = {e.canonical_name for e in result.entities}
        assert "United States Senate" in names
        assert "Outbound Investment Transparency Act" in names

        # Check relationship
        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.source == "United States Senate"
        assert rel.target == "Outbound Investment Transparency Act"
        assert rel.relationship_type == "PASSED"

        # Verify mock called with proper parameters
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "mock-model"
        assert "chunk_text" in call_kwargs["contents"] or chunk_text in call_kwargs["contents"]

    def test_empty_chunk_text_skips_llm_call(self):
        mock_client = MagicMock()
        service = GraphExtractorService(client=mock_client)

        result = service.extract_from_chunk(
            chunk_id=1,
            document_id=1,
            chunk_text="   ",
        )

        assert result.status == "skipped_empty"
        assert result.entities == []
        assert result.relationships == []
        mock_client.models.generate_content.assert_not_called()

    def test_malformed_llm_output_handled_gracefully(self):
        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        service = GraphExtractorService(client=mock_client)

        result = service.extract_from_chunk(
            chunk_id=2,
            document_id=1,
            chunk_text="Some valid chunk text.",
        )

        assert result.status == "error"
        assert len(result.validation_errors) > 0
        assert "Failed to parse LLM response as JSON" in result.validation_errors[0]
        assert result.entities == []
        assert result.relationships == []

    def test_llm_api_exception_handled_without_crashing(self):
        mock_client = MagicMock()
        fake_key = "AIza" + "Sy" + "FakeKey1234567890123456789012345"
        mock_client.models.generate_content.side_effect = RuntimeError(f"API rate limit exceeded {fake_key}")

        service = GraphExtractorService(client=mock_client)

        result = service.extract_from_chunk(
            chunk_id=3,
            document_id=1,
            chunk_text="Some text.",
        )

        assert result.status == "error"
        assert len(result.validation_errors) > 0
        # Check that API key was sanitized
        assert "[REDACTED_API_KEY]" in result.validation_errors[0]
        assert fake_key not in result.validation_errors[0]


    def test_dual_model_alternation(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({"entities": [], "relationships": []})

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        service = GraphExtractorService(
            client=mock_client,
            models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )

        r1 = service.extract_from_chunk(1, 1, "Chunk 1 content")
        r2 = service.extract_from_chunk(2, 1, "Chunk 2 content")
        r3 = service.extract_from_chunk(3, 1, "Chunk 3 content")
        r4 = service.extract_from_chunk(4, 1, "Chunk 4 content")

        assert r1.model_used == "gemini-3.5-flash-lite"
        assert r2.model_used == "gemini-3.1-flash-lite"
        assert r3.model_used == "gemini-3.5-flash-lite"
        assert r4.model_used == "gemini-3.1-flash-lite"

        calls = mock_client.models.generate_content.call_args_list
        assert len(calls) == 4
        assert calls[0][1]["model"] == "gemini-3.5-flash-lite"
        assert calls[1][1]["model"] == "gemini-3.1-flash-lite"
        assert calls[2][1]["model"] == "gemini-3.5-flash-lite"
        assert calls[3][1]["model"] == "gemini-3.1-flash-lite"

    def test_model_rate_pacer_intervals(self):
        from app.graph_extractor import ModelRatePacer

        pacer = ModelRatePacer(max_rpm_per_model=60.0, combined_max_rpm=120.0)
        assert pacer.min_model_interval == 1.0
        assert pacer.min_combined_interval == 0.5

        # Slot 1: immediate
        w1 = pacer.wait_for_slot(model="m1")
        assert w1 == 0.0

        # Slot 2 for different model: respects combined interval (0.5s)
        # Note: since time hasn't passed, wait is ~0.5s
        with pacer._lock:
            time_now = pacer._last_overall_call
            assert time_now > 0

