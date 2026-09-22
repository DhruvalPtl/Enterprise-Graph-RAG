"""
Graph RAG Phase G2.1: Entity and Relationship Ontology, Canonicalization, and Structured Extraction Contract.

Defines:
- 15 Initial Entity Types
- 19 Controlled Relationship Types
- Deterministic Entity Canonicalization
- Pydantic Extraction Schemas with Provenance and Validation
"""
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# 1. ENTITY ONTOLOGY (15 Types)
# ==============================================================================

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    MODEL = "MODEL"
    ALGORITHM = "ALGORITHM"
    DATASET = "DATASET"
    CONCEPT = "CONCEPT"
    METHOD = "METHOD"
    METRIC = "METRIC"
    EVENT = "EVENT"
    REGULATION = "REGULATION"
    POLICY = "POLICY"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    DATE = "DATE"


VALID_ENTITY_TYPES: Set[str] = {t.value for t in EntityType}

ENTITY_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "PERSON": "Named individuals, researchers, authors, executives, or public figures (e.g. 'Geoffrey Hinton', 'Daniel Jurafsky').",
    "ORGANIZATION": "Companies, academic institutions, government bodies, standards groups, or committees (e.g. 'Stanford University', 'U.S. Senate', 'Google').",
    "TECHNOLOGY": "General hardware, software frameworks, protocols, or infrastructure (e.g. 'PyTorch', 'Docker', 'PostgreSQL', 'GPU').",
    "MODEL": "Specific machine learning models or neural architectures (e.g. 'wav2vec 2.0', 'BERT', 'GPT-4', 'MiniLM').",
    "ALGORITHM": "Specific computational or statistical procedures (e.g. 'Connectionist Temporal Classification', 'Beam Search', 'Gradient Descent').",
    "DATASET": "Named corpora, benchmarks, or datasets (e.g. 'ImageNet', 'Librispeech', 'GLUE', 'SQuAD').",
    "CONCEPT": "Theoretical paradigms, abstractions, or domain concepts (e.g. 'Self-Attention', 'Overfitting', 'Quantization').",
    "METHOD": "Techniques, training methodologies, or operational protocols (e.g. 'Reinforcement Learning from Human Feedback', 'Cross-Validation').",
    "METRIC": "Evaluation measures, loss functions, or quantitative indicators (e.g. 'Word Error Rate', 'BLEU', 'F1 Score', 'CTC Loss').",
    "EVENT": "Named occurrences, summits, conferences, or historical events (e.g. 'White House AI Commitments', 'NeurIPS 2023').",
    "REGULATION": "Legislation, statutory acts, executive orders, or legal mandates (e.g. 'Outbound Investment Transparency Act', 'EU AI Act').",
    "POLICY": "Institutional guidelines, corporate rules, standards, or access policies (e.g. 'Enterprise Access Control Policy', 'Acceptable Use Policy').",
    "PRODUCT": "Commercial software, platforms, tools, or physical goods (e.g. 'FastAPI', 'Acrobat', 'Enterprise Search Platform').",
    "LOCATION": "Geopolitical entities, countries, cities, or facilities (e.g. 'United States', 'European Union', 'Washington D.C.').",
    "DATE": "Specific calendar dates, timeline markers, or years (e.g. 'July 25, 2023', '2024').",
}


# ==============================================================================
# 2. RELATIONSHIP ONTOLOGY (19 Controlled Types)
# ==============================================================================

class RelationshipType(str, Enum):
    USES = "USES"
    HAS_COMPONENT = "HAS_COMPONENT"
    PART_OF = "PART_OF"
    DEVELOPED_BY = "DEVELOPED_BY"
    CREATED_BY = "CREATED_BY"
    PROPOSED_BY = "PROPOSED_BY"
    IMPLEMENTED_BY = "IMPLEMENTED_BY"
    EVALUATED_ON = "EVALUATED_ON"
    ACHIEVES = "ACHIEVES"
    IMPROVES = "IMPROVES"
    COMPARES_WITH = "COMPARES_WITH"
    RELATED_TO = "RELATED_TO"
    LOCATED_IN = "LOCATED_IN"
    OCCURRED_ON = "OCCURRED_ON"
    PASSED = "PASSED"
    SIGNED = "SIGNED"
    GOVERNED_BY = "GOVERNED_BY"
    APPLIES_TO = "APPLIES_TO"
    SUPPORTS = "SUPPORTS"


VALID_RELATIONSHIP_TYPES: Set[str] = {t.value for t in RelationshipType}

RELATIONSHIP_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "USES": "Subject employs or utilizes Object as a tool, dependency, or mechanism.",
    "HAS_COMPONENT": "Subject contains or encompasses Object as a sub-component or layer.",
    "PART_OF": "Subject is a sub-element or constituent of Object (inverse of HAS_COMPONENT).",
    "DEVELOPED_BY": "Subject was engineered, designed, or built by Object (person or organization).",
    "CREATED_BY": "Subject was authored, invented, or brought into existence by Object.",
    "PROPOSED_BY": "Subject (concept, method, legislation) was introduced or drafted by Object.",
    "IMPLEMENTED_BY": "Subject was executed, programmed, or enforced by Object.",
    "EVALUATED_ON": "Subject (model, method) was benchmarked or tested against Object (dataset, metric).",
    "ACHIEVES": "Subject attains or records Object (a specific metric, milestone, or score).",
    "IMPROVES": "Subject advances or increases the quality/performance of Object.",
    "COMPARES_WITH": "Subject is directly evaluated or benchmarked against Object.",
    "RELATED_TO": "Subject has a general, explicit domain association with Object.",
    "LOCATED_IN": "Subject is situated or headquartered in Object (location).",
    "OCCURRED_ON": "Subject (event, act, meeting) transpired on Object (date/time).",
    "PASSED": "Subject (legislative body) enacted or approved Object (regulation or bill).",
    "SIGNED": "Subject entered into or endorsed Object (treaty, pledge, commitment, or policy).",
    "GOVERNED_BY": "Subject is regulated, constrained, or overseen by Object (policy, regulation).",
    "APPLIES_TO": "Subject (policy, regulation, rule) targets or governs Object.",
    "SUPPORTS": "Subject provides backing, infrastructure, evidence, or assistance to Object.",
}


# ==============================================================================
# 3. DETERMINISTIC ENTITY CANONICALIZATION
# ==============================================================================

# Common unambiguous acronym expansions
ACRONYM_MAP: Dict[str, str] = {
    "u.s.": "United States",
    "u.s": "United States",
    "us": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "eu": "European Union",
    "e.u.": "European Union",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "ai": "Artificial Intelligence",
    "nlp": "Natural Language Processing",
    "asr": "Automatic Speech Recognition",
    "ctc": "Connectionist Temporal Classification",
}

# Month name mapping for date normalization
MONTH_EXPANSIONS: Dict[str, str] = {
    "jan.": "January", "jan": "January",
    "feb.": "February", "feb": "February",
    "mar.": "March", "mar": "March",
    "apr.": "April", "apr": "April",
    "may": "May",
    "jun.": "June", "jun": "June",
    "jul.": "July", "jul": "July",
    "aug.": "August", "aug": "August",
    "sep.": "September", "sept.": "September", "sep": "September",
    "oct.": "October", "oct": "October",
    "nov.": "November", "nov": "November",
    "dec.": "December", "dec": "December",
}


def normalize_date_string(text: str) -> str:
    """Normalizes date strings to canonical representations (e.g. 'Jul. 25, 2023' -> 'July 25, 2023')."""
    date_pat = re.compile(
        r"\b(January|Jan\.?|February|Feb\.?|March|Mar\.?|April|Apr\.?|May|June|Jun\.?|July|Jul\.?|August|Aug\.?|September|Sept?\.?|October|Oct\.?|November|Nov\.?|December|Dec\.?)\s*(\d{1,2}),?\s*(\d{4})",
        re.IGNORECASE,
    )
    match = date_pat.search(text)
    if match:
        raw_m, day, year = match.groups()
        norm_m = MONTH_EXPANSIONS.get(raw_m.lower().strip(), raw_m.capitalize().rstrip("."))
        return f"{norm_m} {int(day)}, {year}"
    return text


def canonicalize_entity(name: str, entity_type: str) -> str:
    """
    Deterministically normalizes an entity surface mention into a canonical form.
    
    Principles:
    - Precision over recall: do not aggressively conflate entities unless clearly equivalent.
    - Trim surrounding whitespace and punctuation quotes.
    - Normalize interior whitespace runs.
    - Remove leading grammatical articles ('the ', 'a ', 'an ') if not part of a fixed name.
    - Expand unambiguous geopolitical abbreviations (e.g. 'U.S. Senate' -> 'United States Senate').
    - Standardize dates.
    """
    if not name or not name.strip():
        return ""

    # Strip enclosing quotes, brackets, and whitespace
    clean = name.strip().strip("'\"`’“”()[]{}")
    # Normalize internal whitespace
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean:
        return ""

    # Specific date normalization
    if entity_type.upper() == "DATE":
        return normalize_date_string(clean)

    # Remove leading English articles (case-insensitive)
    for article in ("the ", "a ", "an "):
        if clean.lower().startswith(article) and len(clean) > len(article) + 2:
            clean = clean[len(article):].strip()
            break

    # Normalize common prefix/acronym variants cleanly
    # Geopolitical expansions
    us_pattern = re.compile(r"^(?:u\.s\.a\.|usa|u\.s\.|u\.s\b|us\b)\s*", re.IGNORECASE)
    if us_pattern.match(clean):
        rest = us_pattern.sub("", clean).strip()
        clean = f"United States {rest}".strip() if rest else "United States"

    eu_pattern = re.compile(r"^(?:e\.u\.|eu\b)\s*", re.IGNORECASE)
    if eu_pattern.match(clean):
        rest = eu_pattern.sub("", clean).strip()
        clean = f"European Union {rest}".strip() if rest else "European Union"

    uk_pattern = re.compile(r"^(?:u\.k\.|uk\b)\s*", re.IGNORECASE)
    if uk_pattern.match(clean):
        rest = uk_pattern.sub("", clean).strip()
        clean = f"United Kingdom {rest}".strip() if rest else "United Kingdom"

    return clean


# ==============================================================================
# 4. PYDANTIC EXTRACTION SCHEMAS
# ==============================================================================

ENTITY_TYPE_ALIASES: Dict[str, str] = {
    "REPORT": "DATASET",
    "PROJECT": "ORGANIZATION",
    "PAPER": "DATASET",
    "STUDY": "DATASET",
    "SURVEY": "DATASET",
    "DOCUMENT": "DATASET",
    "PUBLICATION": "DATASET",
    "ARTICLE": "DATASET",
    "SYSTEM": "TECHNOLOGY",
    "TOOL": "PRODUCT",
    "FRAMEWORK": "TECHNOLOGY",
    "HARDWARE": "TECHNOLOGY",
    "SOFTWARE": "TECHNOLOGY",
    "LIBRARY": "TECHNOLOGY",
    "ARCHITECTURE": "TECHNOLOGY",
    "STANDARD": "REGULATION",
    "LAW": "REGULATION",
    "LEGISLATION": "REGULATION",
    "ACT": "REGULATION",
    "BILL": "REGULATION",
    "DIRECTIVE": "REGULATION",
    "INITIATIVE": "POLICY",
    "PROGRAM": "ORGANIZATION",
    "COMPANY": "ORGANIZATION",
    "INSTITUTION": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "AGENCY": "ORGANIZATION",
    "CORP": "ORGANIZATION",
    "INDIVIDUAL": "PERSON",
    "RESEARCHER": "PERSON",
    "AUTHOR": "PERSON",
    "TASK": "CONCEPT",
    "POSITION": "CONCEPT",
    "ROLE": "CONCEPT",
    "BENCHMARK": "DATASET",
    "METRIC_SCORE": "METRIC",
    "SCORE": "METRIC",
    "HAS_COMPONENT": "CONCEPT",
}

RELATIONSHIP_TYPE_ALIASES: Dict[str, str] = {
    "USED_BY": "USES",
    "DESCRIBED_BY": "RELATED_TO",
    "COMPUTES": "USES",
    "REPLACES": "IMPROVES",
    "EVALUATED_BY": "EVALUATED_ON",
    "TESTED_ON": "EVALUATED_ON",
    "BENCHMARKED_ON": "EVALUATED_ON",
    "BUILT_BY": "DEVELOPED_BY",
    "MADE_BY": "DEVELOPED_BY",
    "FOUNDED_BY": "CREATED_BY",
    "INVENTED_BY": "CREATED_BY",
    "AUTHORED_BY": "CREATED_BY",
    "AUTHORED": "CREATED_BY",
    "PUBLISHED_BY": "CREATED_BY",
    "PRODUCED_BY": "DEVELOPED_BY",
    "CONTAINED_IN": "PART_OF",
    "CONTAINS": "HAS_COMPONENT",
    "SUBSET_OF": "PART_OF",
    "ASSOCIATED_WITH": "RELATED_TO",
    "SIMILAR_TO": "COMPARES_WITH",
}


class EntityExtraction(BaseModel):
    """Represents a single extracted entity candidate from a text chunk."""
    name: str = Field(..., description="Exact or surface name of the entity as mentioned in the text.")
    canonical_name: Optional[str] = Field(None, description="Canonical normalized form of the entity.")
    entity_type: str = Field(..., description="Entity category from the controlled ontology.")
    description: Optional[str] = Field(None, description="Optional brief qualifying context from the chunk.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score.")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name_not_empty(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Entity name cannot be empty.")
        return v.strip()

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_entity_type(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("entity_type must be a string.")
        norm_type = v.strip().upper()
        if norm_type in ENTITY_TYPE_ALIASES:
            norm_type = ENTITY_TYPE_ALIASES[norm_type]
        if norm_type not in VALID_ENTITY_TYPES:
            raise ValueError(f"Invalid entity_type '{v}'. Must be one of: {sorted(list(VALID_ENTITY_TYPES))}")
        return norm_type


class RelationshipExtraction(BaseModel):
    """Represents a directed relationship between two entities extracted from a text chunk."""
    source: str = Field(..., description="Name or canonical name of the source entity.")
    target: str = Field(..., description="Name or canonical name of the target entity.")
    relationship_type: str = Field(..., description="Relationship category from the controlled ontology.")
    evidence: Optional[str] = Field(None, description="Exact phrase or sentence supporting this relationship.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score.")

    @field_validator("source", "target", mode="before")
    @classmethod
    def validate_endpoints_not_empty(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Relationship endpoint cannot be empty.")
        return v.strip()

    @field_validator("relationship_type", mode="before")
    @classmethod
    def validate_relationship_type(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("relationship_type must be a string.")
        norm_type = v.strip().upper()
        if norm_type in RELATIONSHIP_TYPE_ALIASES:
            norm_type = RELATIONSHIP_TYPE_ALIASES[norm_type]
        if norm_type not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(f"Invalid relationship_type '{v}'. Must be one of: {sorted(list(VALID_RELATIONSHIP_TYPES))}")
        return norm_type


class RawExtractionOutput(BaseModel):
    """Raw structured output format expected from the LLM."""
    entities: List[EntityExtraction] = Field(default_factory=list, description="List of extracted entities.")
    relationships: List[RelationshipExtraction] = Field(default_factory=list, description="List of extracted relationships.")


class ValidatedExtractionResult(BaseModel):
    """
    Validated extraction result ready for downstream persistence or analysis.
    Retains full chunk provenance and tracks deduplication and dropped invalid items.
    """
    document_id: int
    chunk_id: int
    page_number: Optional[int] = None
    entities: List[EntityExtraction] = Field(default_factory=list)
    relationships: List[RelationshipExtraction] = Field(default_factory=list)
    dropped_entities: List[Dict[str, Any]] = Field(default_factory=list)
    dropped_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    status: str = "success"
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "entities": [e.model_dump() for e in self.entities],
            "relationships": [r.model_dump() for r in self.relationships],
            "dropped_entities": self.dropped_entities,
            "dropped_relationships": self.dropped_relationships,
            "validation_errors": self.validation_errors,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_used": self.model_used,
        }


# ==============================================================================
# 5. VALIDATION & SANITIZATION PIPELINE
# ==============================================================================

def validate_and_sanitize_extraction(
    raw_output: RawExtractionOutput,
    document_id: int,
    chunk_id: int,
    page_number: Optional[int] = None,
) -> ValidatedExtractionResult:
    """
    Validates, deduplicates, and sanitizes raw extraction output against the ontology.
    
    Enforces:
    1. Entity deduplication: merges entities with identical (canonical_name, entity_type).
    2. Endpoint resolution: ensures every relationship's source and target resolves to an extracted entity.
    3. Self-referential rejection: drops relationships where canonical source equals canonical target.
    4. Relationship deduplication: drops duplicate triples (canonical_source, canonical_target, relationship_type).
    5. Provenance attachment: tags with document_id, chunk_id, and page_number.
    """
    result = ValidatedExtractionResult(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=page_number,
    )

    # 1. Process and deduplicate entities
    # Map from surface names / variations to the canonical entity
    entity_lookup: Dict[str, EntityExtraction] = {}
    unique_entities: Dict[Tuple[str, str], EntityExtraction] = {}

    for ent in raw_output.entities:
        if not ent.name or not ent.name.strip():
            result.dropped_entities.append({"entity": ent.model_dump(), "reason": "empty_name"})
            continue

        # Canonicalize if needed
        canon = ent.canonical_name or canonicalize_entity(ent.name, ent.entity_type)
        if not canon:
            canon = ent.name.strip()
        ent.canonical_name = canon

        key = (canon.lower(), ent.entity_type)
        if key in unique_entities:
            existing = unique_entities[key]
            # Merge descriptions or preserve higher confidence
            if ent.confidence > existing.confidence:
                existing.confidence = ent.confidence
            if not existing.description and ent.description:
                existing.description = ent.description
            # Register surface name for lookup
            entity_lookup[ent.name.strip().lower()] = existing
            result.dropped_entities.append({
                "entity": ent.model_dump(),
                "reason": f"duplicate_of_{existing.canonical_name}",
            })
        else:
            unique_entities[key] = ent
            entity_lookup[ent.name.strip().lower()] = ent
            entity_lookup[canon.strip().lower()] = ent

    result.entities = list(unique_entities.values())

    # 2. Process and validate relationships
    seen_relationships: Set[Tuple[str, str, str]] = set()

    for rel in raw_output.relationships:
        src_raw = rel.source.strip().lower()
        tgt_raw = rel.target.strip().lower()

        src_ent = entity_lookup.get(src_raw)
        tgt_ent = entity_lookup.get(tgt_raw)

        # Validate that endpoints resolve to extracted entities
        if not src_ent:
            result.dropped_relationships.append({
                "relationship": rel.model_dump(),
                "reason": f"source_entity_not_found: '{rel.source}'",
            })
            continue

        if not tgt_ent:
            result.dropped_relationships.append({
                "relationship": rel.model_dump(),
                "reason": f"target_entity_not_found: '{rel.target}'",
            })
            continue

        # Prevent self-loops
        if src_ent.canonical_name.lower() == tgt_ent.canonical_name.lower():
            result.dropped_relationships.append({
                "relationship": rel.model_dump(),
                "reason": f"self_referential_cycle: '{src_ent.canonical_name}'",
            })
            continue

        # Normalize endpoints to canonical names
        rel_key = (
            src_ent.canonical_name.lower(),
            tgt_ent.canonical_name.lower(),
            rel.relationship_type.upper(),
        )

        if rel_key in seen_relationships:
            result.dropped_relationships.append({
                "relationship": rel.model_dump(),
                "reason": "duplicate_relationship",
            })
            continue

        seen_relationships.add(rel_key)

        # Update relationship endpoints to point to canonical names
        clean_rel = RelationshipExtraction(
            source=src_ent.canonical_name,
            target=tgt_ent.canonical_name,
            relationship_type=rel.relationship_type.upper(),
            evidence=rel.evidence,
            confidence=rel.confidence,
        )
        result.relationships.append(clean_rel)

    return result
