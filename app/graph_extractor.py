"""
Graph RAG Phase G2.1: LLM Extraction Service.

Extracts structured entities and controlled relationships from document chunks using Gemini,
enforces ontology schemas, runs deterministic canonicalization, and validates provenance.
"""
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

from app.config import GEMINI_API_KEY, GEMINI_LLM_MODEL
from app.graph_ontology import (
    RawExtractionOutput,
    ValidatedExtractionResult,
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
    ENTITY_TYPE_DESCRIPTIONS,
    RELATIONSHIP_TYPE_DESCRIPTIONS,
    validate_and_sanitize_extraction,
)

logger = logging.getLogger("graph_extractor")


# System prompt defining strict factual extraction rules and controlled ontology
EXTRACTION_SYSTEM_PROMPT = f"""You are a precise factual knowledge graph extraction engine for enterprise RAG.
Your task is to extract entities and directed relationships exclusively from the provided text chunk.

STRICT EXTRACTION RULES:
1. FACTUAL GROUNDING: Extract only facts explicitly stated in the chunk text. Do NOT assume, infer, or use outside world knowledge.
2. CONTROLLED ENTITY TYPES: You MUST assign each entity exactly one of these 15 types:
{json.dumps(ENTITY_TYPE_DESCRIPTIONS, indent=2)}

3. CONTROLLED RELATIONSHIP TYPES: You MUST assign each relationship exactly one of these 19 types:
{json.dumps(RELATIONSHIP_TYPE_DESCRIPTIONS, indent=2)}

4. NO INVENTED RELATIONS: If a relationship cannot be represented accurately using one of the 19 controlled types, OMIT IT. Do NOT invent new relation names.
5. REFERENTIAL INTEGRITY: Every relationship's 'source' and 'target' MUST match the 'name' of an entity defined in your 'entities' list.
6. NO SELF-LOOPS: The 'source' and 'target' of a relationship must not be the same entity.
7. ATOMIC ENTITIES: Extract distinct, well-defined entities (e.g. 'Outbound Investment Transparency Act', 'U.S. Senate'). Avoid long full sentences as entity names.
8. RETURN STRUCTURED JSON: You must return valid JSON matching the requested schema.
"""


def sanitize_error_message(msg: str) -> str:
    """Removes any accidental API keys from exception strings."""
    return re.sub(r"AIza[0-9A-Za-z-_]{25,45}", "[REDACTED_API_KEY]", msg)


def parse_raw_extraction_json(text: str) -> RawExtractionOutput:
    """
    Parses and validates raw JSON output from the LLM into RawExtractionOutput.
    Handles optional markdown code fences or surrounding whitespace.
    """
    if not text or not text.strip():
        return RawExtractionOutput(entities=[], relationships=[])

    clean_text = text.strip()
    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
    if fence_match:
        clean_text = fence_match.group(1).strip()

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {exc}. Response snippet: {clean_text[:200]}")

    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got {type(data).__name__}")

    return RawExtractionOutput.model_validate(data)


class ModelRatePacer:
    """
    Thread-safe dual-model rate limiter and request pacer.
    Guarantees:
      1. No individual model on an account exceeds max_rpm_per_model (e.g. 14.5 RPM -> 4.14s min interval).
      2. Combined rate across all models does not exceed combined_max_rpm (e.g. 28.5 RPM -> 2.105s min interval).
    """

    def __init__(self, max_rpm_per_model: float = 14.5, combined_max_rpm: float = 28.5):
        self.max_rpm_per_model = max_rpm_per_model
        self.combined_max_rpm = combined_max_rpm
        self.min_model_interval = (60.0 / max_rpm_per_model) if max_rpm_per_model > 0 else 0.0
        self.min_combined_interval = (60.0 / combined_max_rpm) if combined_max_rpm > 0 else 0.0
        self._last_call_per_model: Dict[str, float] = {}
        self._last_overall_call: float = 0.0
        self._lock = threading.Lock()

    def wait_for_slot(self, model: str, account_id: Optional[int] = None) -> float:
        """
        Calculates required wait time to honor both per-model and combined intervals,
        reserves the scheduled slot atomically, and sleeps if needed.
        Returns the duration slept in seconds.
        """
        with self._lock:
            key = f"{account_id}:{model}" if account_id is not None else model
            now = time.time()

            # 1. Combined interval pacing
            time_since_last_overall = now - self._last_overall_call
            wait_overall = max(0.0, self.min_combined_interval - time_since_last_overall)

            # 2. Per-model interval pacing on this account
            last_model_time = self._last_call_per_model.get(key, 0.0)
            time_since_last_model = now - last_model_time
            wait_model = max(0.0, self.min_model_interval - time_since_last_model)

            wait_seconds = max(wait_overall, wait_model)
            target_time = now + wait_seconds

            # Reserve slot timestamp
            self._last_overall_call = target_time
            self._last_call_per_model[key] = target_time

        if wait_seconds > 0.005:
            end_time = time.time() + wait_seconds
            while time.time() < end_time:
                time.sleep(min(0.2, max(0.01, end_time - time.time())))

        return wait_seconds


class GraphExtractorService:
    """
    Structured extraction service interfacing with Google Gemini to extract
    knowledge graph entities and relationships with chunk provenance.
    Supports dual-model interleaved execution (e.g. gemini-3.5-flash-lite + gemini-3.1-flash-lite),
    multi-account key pool with round-robin rotation, per-model rate limit pacing, and
    automatic rate-limit cooldown.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        models: Optional[List[str]] = None,
        client: Optional[Any] = None,
        key_pool: Optional[Any] = None,
        pacer: Optional[ModelRatePacer] = None,
    ):
        if models:
            self.models = list(models)
        elif model_name and "," in model_name:
            self.models = [m.strip() for m in model_name.split(",") if m.strip()]
        elif model_name:
            self.models = [model_name]
        else:
            env_model = os.getenv("GEMINI_LLM_MODEL")
            if env_model:
                self.models = [env_model]
            else:
                self.models = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

        self.model_name = self.models[0]
        self.fallback_model = self.models[1] if len(self.models) > 1 else "gemini-3.1-flash-lite"
        self.last_used_model: str = self.models[0]
        self._turn_index: int = 0
        self._lock = threading.Lock()
        self.pacer: Optional[ModelRatePacer] = pacer
        self._api_key = api_key
        self.client = client
        self.key_pool = key_pool

        # If no client or key_pool explicitly provided, try to load key pool from api_key.text
        if self.client is None and self.key_pool is None:
            try:
                from app.key_pool import get_key_pool, DEFAULT_KEY_FILE
                if DEFAULT_KEY_FILE.exists():
                    pool = get_key_pool(DEFAULT_KEY_FILE)
                    if pool.total_accounts > 0:
                        self.key_pool = pool
            except Exception as exc:
                logger.warning(f"Could not load key pool: {exc}")

        # If still no client and no key_pool, fall back to single client via GEMINI_API_KEY
        if self.client is None and self.key_pool is None:
            resolved_key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
            if not resolved_key or not resolved_key.strip():
                raise ValueError(
                    "Gemini API key is required for Graph Extraction. "
                    "Please provide api_key.text or set the GEMINI_API_KEY environment variable."
                )

            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "The 'google-genai' SDK is required for Gemini extraction. "
                    "Please install it via: pip install google-genai"
                )

            self.client = genai.Client(api_key=resolved_key.strip())

    def extract_from_chunk(
        self,
        chunk_id: int,
        document_id: int,
        chunk_text: str,
        page_number: Optional[int] = None,
        temperature: float = 0.0,
    ) -> ValidatedExtractionResult:
        """
        Extracts entities and relationships from a text chunk, validates them against
        the controlled ontology, deduplicates entities and edges, and attaches chunk provenance.
        Rotates between dual models per request, honors per-model rate limits, and automatically
        shifts accounts when all models on an account are exhausted.
        """
        if not chunk_text or not chunk_text.strip():
            return ValidatedExtractionResult(
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page_number,
                status="skipped_empty",
            )

        prompt = f"""Extract knowledge graph entities and relationships from the following text chunk:

--- CHUNK TEXT ---
{chunk_text.strip()}
--- END CHUNK TEXT ---
"""

        # Determine preferred starting model for this chunk turn
        with self._lock:
            turn = self._turn_index
            self._turn_index += 1

        preferred_idx = turn % len(self.models)
        preferred_model = self.models[preferred_idx]
        candidate_models = [preferred_model] + [m for m in self.models if m != preferred_model]
        if self.fallback_model and self.fallback_model not in candidate_models:
            candidate_models.append(self.fallback_model)

        max_attempts = max(6, (self.key_pool.total_accounts * len(candidate_models) * 2)) if self.key_pool else 3
        attempt = 0
        last_error = None

        while attempt < max_attempts:
            attempt += 1

            if self.key_pool:
                active_client, account_info = self.key_pool.get_client_and_account()
            else:
                active_client = self.client
                account_info = None

            # Determine ready models for the active account
            if account_info:
                ready_models = account_info.get_ready_models(candidate_models)
                if not ready_models:
                    # All candidate models on this account are on cooldown! Shift to next account.
                    logger.warning(
                        f"All models in cooldown on account '{account_info.name}'. Shifting account..."
                    )
                    self.key_pool.mark_rate_limited(account_info.account_id, suggested_delay=60.0)
                    continue
                model_to_try = ready_models[0]
            else:
                model_to_try = candidate_models[(attempt - 1) % len(candidate_models)]

            # Pacing slot: ensures per-model <= 14.5 RPM and combined <= 28.5 RPM
            if self.pacer:
                acc_id = account_info.account_id if account_info else None
                self.pacer.wait_for_slot(model=model_to_try, account_id=acc_id)

            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=RawExtractionOutput,
                )

                response = active_client.models.generate_content(
                    model=model_to_try,
                    contents=prompt,
                    config=config,
                )

                if not response or not getattr(response, "text", None):
                    raise RuntimeError("Empty response received from Gemini API.")

                raw_output = parse_raw_extraction_json(response.text)
                self.model_name = model_to_try
                self.last_used_model = model_to_try

                # Validate, deduplicate, and attach provenance
                validated = validate_and_sanitize_extraction(
                    raw_output=raw_output,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page_number=page_number,
                )
                validated.model_used = model_to_try
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    validated.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    validated.output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                return validated

            except Exception as exc:
                last_error = exc
                err_str = str(exc)

                # 0. Account Disabled / Unauthorized (401 / UNAUTHENTICATED)
                is_unauthenticated = any(k in err_str for k in ("401", "UNAUTHENTICATED", "disabled", "deleted"))
                if is_unauthenticated:
                    if self.key_pool and account_info:
                        self.key_pool.mark_disabled(account_info.account_id, reason="401 Disabled/Unauthenticated")
                        continue

                # 1. Rate Limit (429 / RESOURCE_EXHAUSTED)
                is_rate_limit = any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED"))
                if is_rate_limit:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if not match:
                        match = re.search(r"retryDelay': '(\d+)s'", err_str)
                    delay = (float(match.group(1)) + 1.0) if match else 60.0

                    is_daily = any(k in err_str.lower() for k in ("per day", "perday", "day quota", "limit: 500"))
                    if is_daily:
                        delay = max(delay, 3600.0)

                    if account_info:
                        account_info.mark_model_cooldown(model_to_try, delay=delay)
                        logger.warning(
                            f"Model {model_to_try} hit rate limit on account '{account_info.name}' (delay {delay:.0f}s). "
                            f"Checking remaining ready models on this account..."
                        )
                        # If all candidate models on this account are now on cooldown, mark account rate limited
                        if not account_info.get_ready_models(candidate_models):
                            if self.key_pool:
                                self.key_pool.mark_rate_limited(account_info.account_id, suggested_delay=delay)
                        continue
                    else:
                        time.sleep(min(delay, 60.0))
                        continue

                # 2. Server Temporary Overload / 503
                is_server_unavailable = any(k in err_str for k in ("503", "UNAVAILABLE", "high demand", "overloaded"))
                if is_server_unavailable:
                    if self.key_pool and account_info:
                        self.key_pool.mark_rate_limited(account_info.account_id, suggested_delay=10.0)
                        continue
                    else:
                        time.sleep(2.0 * attempt)
                        continue

                # 3. Network Connection Glitch
                is_network_error = any(k in err_str for k in ("11001", "getaddrinfo", "ConnectError", "timeout", "timed out"))
                if is_network_error and attempt < max_attempts:
                    time.sleep(1.5 * attempt)
                    continue

                # 4. Model Unsupported / 404
                is_model_error = any(k in err_str for k in ("404", "NOT_FOUND", "not supported"))
                if is_model_error:
                    logger.warning(f"Model {model_to_try} returned 404/not supported. Removing from candidates.")
                    if model_to_try in candidate_models:
                        candidate_models.remove(model_to_try)
                    if not candidate_models:
                        break
                    continue

                # 5. Schema / Parsing error: retry once
                if attempt < 2:
                    time.sleep(1.0)
                    continue
                break

        # If all attempts fail, log and return safe empty result with error status
        err_msg = sanitize_error_message(str(last_error)) if last_error else "Unknown extraction failure"
        logger.error(f"Extraction failed for chunk {chunk_id} (doc {document_id}): {err_msg}")
        return ValidatedExtractionResult(
            document_id=document_id,
            chunk_id=chunk_id,
            page_number=page_number,
            status="error",
            validation_errors=[err_msg],
        )
