"""
Unit tests for Graph RAG Phase G2.1 Ontology, Canonicalization, and Structured Extraction Contract.
"""
import pytest
from pydantic import ValidationError

from app.graph_ontology import (
    EntityType,
    RelationshipType,
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES,
    canonicalize_entity,
    normalize_date_string,
    EntityExtraction,
    RelationshipExtraction,
    RawExtractionOutput,
    ValidatedExtractionResult,
    validate_and_sanitize_extraction,
)


class TestGraphOntologyDefinitions:
    """Validates the 15 entity types and 19 relationship types."""

    def test_entity_types_count_and_members(self):
        assert len(EntityType) == 15
        expected = {
            "PERSON", "ORGANIZATION", "TECHNOLOGY", "MODEL", "ALGORITHM",
            "DATASET", "CONCEPT", "METHOD", "METRIC", "EVENT",
            "REGULATION", "POLICY", "PRODUCT", "LOCATION", "DATE"
        }
        assert set(EntityType._member_names_) == expected
        assert VALID_ENTITY_TYPES == expected

    def test_relationship_types_count_and_members(self):
        assert len(RelationshipType) == 19
        expected = {
            "USES", "HAS_COMPONENT", "PART_OF", "DEVELOPED_BY", "CREATED_BY",
            "PROPOSED_BY", "IMPLEMENTED_BY", "EVALUATED_ON", "ACHIEVES",
            "IMPROVES", "COMPARES_WITH", "RELATED_TO", "LOCATED_IN",
            "OCCURRED_ON", "PASSED", "SIGNED", "GOVERNED_BY", "APPLIES_TO", "SUPPORTS"
        }
        assert set(RelationshipType._member_names_) == expected
        assert VALID_RELATIONSHIP_TYPES == expected


class TestCanonicalization:
    """Validates deterministic entity normalization rules."""

    def test_whitespace_and_punctuation_trimming(self):
        assert canonicalize_entity("  Transformer  ", "MODEL") == "Transformer"
        assert canonicalize_entity(' "BERT" ', "MODEL") == "BERT"
        assert canonicalize_entity("`FastAPI`", "PRODUCT") == "FastAPI"
        assert canonicalize_entity("((PyTorch))", "TECHNOLOGY") == "PyTorch"

    def test_leading_article_removal(self):
        assert canonicalize_entity("the United States Senate", "ORGANIZATION") == "United States Senate"
        assert canonicalize_entity("The European Union", "LOCATION") == "European Union"
        assert canonicalize_entity("a convolutional neural network", "CONCEPT") == "convolutional neural network"
        assert canonicalize_entity("an attention layer", "CONCEPT") == "attention layer"

    def test_acronym_and_prefix_normalization(self):
        assert canonicalize_entity("U.S. Senate", "ORGANIZATION") == "United States Senate"
        assert canonicalize_entity("the U.S. Senate", "ORGANIZATION") == "United States Senate"
        assert canonicalize_entity("US Department of Energy", "ORGANIZATION") == "United States Department of Energy"
        assert canonicalize_entity("EU AI Act", "REGULATION") == "European Union AI Act"
        assert canonicalize_entity("UK Government", "ORGANIZATION") == "United Kingdom Government"

    def test_date_normalization(self):
        assert normalize_date_string("Jul. 25, 2023") == "July 25, 2023"
        assert normalize_date_string("July 25, 2023") == "July 25, 2023"
        assert normalize_date_string("Nov 14, 2022") == "November 14, 2022"
        assert canonicalize_entity("Jul. 25, 2023", "DATE") == "July 25, 2023"

    def test_empty_string_handling(self):
        assert canonicalize_entity("", "CONCEPT") == ""
        assert canonicalize_entity("   ", "ORGANIZATION") == ""


class TestPydanticSchemaValidation:
    """Validates schema models, field constraints, and rejection of invalid vocabulary."""

    def test_valid_entity_creation(self):
        e = EntityExtraction(name="wav2vec 2.0", entity_type="MODEL")
        assert e.name == "wav2vec 2.0"
        assert e.entity_type == "MODEL"
        assert e.confidence == 1.0

    def test_invalid_entity_type_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            EntityExtraction(name="Something", entity_type="INVALID_TYPE")
        assert "Invalid entity_type" in str(excinfo.value)

    def test_empty_entity_name_rejected(self):
        with pytest.raises(ValidationError):
            EntityExtraction(name="   ", entity_type="CONCEPT")

    def test_valid_relationship_creation(self):
        r = RelationshipExtraction(
            source="U.S. Senate",
            target="Outbound Investment Transparency Act",
            relationship_type="PASSED",
        )
        assert r.source == "U.S. Senate"
        assert r.target == "Outbound Investment Transparency Act"
        assert r.relationship_type == "PASSED"

    def test_invalid_relationship_type_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            RelationshipExtraction(
                source="A",
                target="B",
                relationship_type="INVENTED_RELATION",
            )
        assert "Invalid relationship_type" in str(excinfo.value)

    def test_empty_relationship_endpoints_rejected(self):
        with pytest.raises(ValidationError):
            RelationshipExtraction(source="", target="B", relationship_type="USES")
        with pytest.raises(ValidationError):
            RelationshipExtraction(source="A", target="  ", relationship_type="USES")


class TestValidationAndSanitizationPipeline:
    """Validates the multi-step sanitization, deduplication, and provenance attachment."""

    def test_successful_validation_and_deduplication(self):
        raw = RawExtractionOutput(
            entities=[
                EntityExtraction(name="U.S. Senate", entity_type="ORGANIZATION"),
                EntityExtraction(name="the U.S. Senate", entity_type="ORGANIZATION"),
                EntityExtraction(name="Outbound Investment Transparency Act", entity_type="REGULATION"),
                EntityExtraction(name="Jul. 25, 2023", entity_type="DATE"),
            ],
            relationships=[
                RelationshipExtraction(
                    source="U.S. Senate",
                    target="Outbound Investment Transparency Act",
                    relationship_type="PASSED",
                ),
                # Duplicate relationship using alternate surface name for source
                RelationshipExtraction(
                    source="the U.S. Senate",
                    target="Outbound Investment Transparency Act",
                    relationship_type="PASSED",
                ),
                RelationshipExtraction(
                    source="Outbound Investment Transparency Act",
                    target="Jul. 25, 2023",
                    relationship_type="OCCURRED_ON",
                ),
            ],
        )

        result = validate_and_sanitize_extraction(
            raw_output=raw,
            document_id=91,
            chunk_id=8273,
            page_number=372,
        )

        assert result.document_id == 91
        assert result.chunk_id == 8273
        assert result.page_number == 372
        assert result.status == "success"

        # 4 raw entities -> 3 unique canonical entities (deduplicated "U.S. Senate" & "the U.S. Senate")
        assert len(result.entities) == 3
        entity_names = {e.canonical_name for e in result.entities}
        assert "United States Senate" in entity_names
        assert "Outbound Investment Transparency Act" in entity_names
        assert "July 25, 2023" in entity_names

        # 3 raw relationships -> 2 unique relationships (1 duplicate dropped)
        assert len(result.relationships) == 2
        rel_signatures = {(r.source, r.relationship_type, r.target) for r in result.relationships}
        assert ("United States Senate", "PASSED", "Outbound Investment Transparency Act") in rel_signatures
        assert ("Outbound Investment Transparency Act", "OCCURRED_ON", "July 25, 2023") in rel_signatures

        # Check dropped tracking
        assert len(result.dropped_entities) == 1
        assert len(result.dropped_relationships) == 1

    def test_unresolved_endpoint_dropped(self):
        raw = RawExtractionOutput(
            entities=[
                EntityExtraction(name="Entity A", entity_type="CONCEPT"),
            ],
            relationships=[
                RelationshipExtraction(
                    source="Entity A",
                    target="Ghost Entity B",
                    relationship_type="USES",
                )
            ],
        )

        result = validate_and_sanitize_extraction(raw, document_id=1, chunk_id=1)
        assert len(result.relationships) == 0
        assert len(result.dropped_relationships) == 1
        assert "target_entity_not_found" in result.dropped_relationships[0]["reason"]

    def test_self_referential_cycle_dropped(self):
        raw = RawExtractionOutput(
            entities=[
                EntityExtraction(name="PyTorch", entity_type="TECHNOLOGY"),
            ],
            relationships=[
                RelationshipExtraction(
                    source="PyTorch",
                    target="PyTorch",
                    relationship_type="USES",
                )
            ],
        )

        result = validate_and_sanitize_extraction(raw, document_id=1, chunk_id=1)
        assert len(result.relationships) == 0
        assert len(result.dropped_relationships) == 1
        assert "self_referential_cycle" in result.dropped_relationships[0]["reason"]

    def test_provenance_and_serialization(self):
        raw = RawExtractionOutput(
            entities=[EntityExtraction(name="Python", entity_type="TECHNOLOGY")],
            relationships=[],
        )
        res = validate_and_sanitize_extraction(raw, document_id=42, chunk_id=100, page_number=5)
        d = res.to_dict()
        assert d["document_id"] == 42
        assert d["chunk_id"] == 100
        assert d["page_number"] == 5
        assert len(d["entities"]) == 1
        assert d["entities"][0]["name"] == "Python"
