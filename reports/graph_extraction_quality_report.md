# Graph RAG Phase G2.1: Extraction Prototype Quality Report

**Execution Date**: 2026-09-18T11:30:00.799038
**Extractor Model**: `gemini-3.5-flash-lite`
**Scope**: Controlled 53-Chunk Prototype across 8 corpus documents

---

## 1. Executive Summary & Quality Metrics

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Chunks Processed** | **53** | Representative sample across all 8 documents |
| **Execution Duration** | **172.3s** | 3.25s per chunk average |
| **Valid Entities Extracted** | **265** | Average 5.0 entities/chunk |
| **Valid Relationships Extracted** | **92** | Average 1.7 relationships/chunk |
| **Duplicate Entities Deduplicated** | **2** | Merged via canonicalization rules |
| **Duplicate/Invalid Relations Dropped** | **0** | Deduplicated or unresolvable endpoints pruned |
| **Extraction Failures** | **1** | Zero fatal crashes |
| **Production DB Status** | **0 rows inserted** | PostgreSQL tables remain completely unpopulated |

---

## 2. Entity Distribution by Controlled Type

| Entity Type | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `ALGORITHM` | 12 | 4.5% | Supported |
| `CONCEPT` | 38 | 14.3% | Supported |
| `DATASET` | 6 | 2.3% | Supported |
| `DATE` | 16 | 6.0% | Supported |
| `EVENT` | 4 | 1.5% | Supported |
| `LOCATION` | 2 | 0.8% | Supported |
| `METHOD` | 13 | 4.9% | Supported |
| `METRIC` | 8 | 3.0% | Supported |
| `MODEL` | 33 | 12.5% | Supported |
| `ORGANIZATION` | 23 | 8.7% | Supported |
| `PERSON` | 75 | 28.3% | Supported |
| `POLICY` | 5 | 1.9% | Supported |
| `PRODUCT` | 5 | 1.9% | Supported |
| `REGULATION` | 4 | 1.5% | Supported |
| `TECHNOLOGY` | 21 | 7.9% | Supported |

---

## 3. Relationship Distribution by Controlled Type

| Relationship Type | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| `ACHIEVES` | 0 | 0.0% | Controlled |
| `APPLIES_TO` | 0 | 0.0% | Controlled |
| `COMPARES_WITH` | 0 | 0.0% | Controlled |
| `CREATED_BY` | 30 | 32.6% | Controlled |
| `DEVELOPED_BY` | 13 | 14.1% | Controlled |
| `EVALUATED_ON` | 10 | 10.9% | Controlled |
| `GOVERNED_BY` | 0 | 0.0% | Controlled |
| `HAS_COMPONENT` | 7 | 7.6% | Controlled |
| `IMPLEMENTED_BY` | 0 | 0.0% | Controlled |
| `IMPROVES` | 0 | 0.0% | Controlled |
| `LOCATED_IN` | 2 | 2.2% | Controlled |
| `OCCURRED_ON` | 2 | 2.2% | Controlled |
| `PART_OF` | 3 | 3.3% | Controlled |
| `PASSED` | 2 | 2.2% | Controlled |
| `PROPOSED_BY` | 1 | 1.1% | Controlled |
| `RELATED_TO` | 12 | 13.0% | Controlled |
| `SIGNED` | 8 | 8.7% | Controlled |
| `SUPPORTS` | 0 | 0.0% | Controlled |
| `USES` | 2 | 2.2% | Controlled |

---

## 4. Exemplar Extracted Subgraphs with Full Provenance

### Example: Chunk 50306 — speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf (Page 206)

#### CHUNK TEXT
```text
198
CHAPTER 8
•
TRANSFORMERS

8.10
Summary

This chapter has introduced the transformer and its components for the language
modeling task introduced in the previous chapter. Here’s a summary of the main
points that we covered:

• Transformers are non-recurrent networks based on multi-head attention,...
```

#### EXTRACTED ENTITIES
| Surface Mention | Canonical Name | Type | Confidence |
| :--- | :--- | :--- | :--- |
| Transformers | **Transformer** | `MODEL` | 0.95 |
| multi-head attention | **Multi-Head Attention** | `CONCEPT` | 0.95 |
| self-attention | **Self-Attention** | `CONCEPT` | 0.95 |
| transformer block | **Transformer Block** | `TECHNOLOGY` | 0.95 |
| multi-head attention layer | **Multi-Head Attention Layer** | `TECHNOLOGY` | 0.95 |

#### EXTRACTED RELATIONSHIPS
| Source Entity | Relationship | Target Entity | Evidence |
| :--- | :--- | :--- | :--- |
| **Transformer** | `HAS_COMPONENT` | **Multi-Head Attention** | *"Transformers are non-recurrent networks based on multi-head attention"* |
| **Multi-Head Attention** | `RELATED_TO` | **Self-Attention** | *"multi-head attention, a kind of self-attention"* |
| **Transformer Block** | `HAS_COMPONENT` | **Multi-Head Attention Layer** | *"These components include a multi-head attention layer"* |

#### PROVENANCE METADATA
- **Document ID**: `96`
- **Chunk ID**: `50306`
- **Page Number**: `206`
- **Filename**: `speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf`

---

### Example: Chunk 50678 — speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf (Page 275)

#### CHUNK TEXT
```text
12.3
•
DETAILS OF THE ENCODER-DECODER MODEL
267

that are too similar, suggesting that they were copies rather than translations). Or
pairs can be ranked by their multilingual embedding cosine score and low-scoring
pairs discarded.

12.3
Details of the Encoder-Decoder Model

|Decoder<br>llegó la bru...
```

#### EXTRACTED ENTITIES
| Surface Mention | Canonical Name | Type | Confidence |
| :--- | :--- | :--- | :--- |
| Encoder-Decoder Model | **Encoder-Decoder Model** | `MODEL` | 0.95 |
| encoder-decoder transformer architecture | **Encoder-Decoder Transformer Architecture** | `MODEL` | 0.95 |
| Figure 12.5 | **Figure 12.5** | `CONCEPT` | 0.9 |

#### EXTRACTED RELATIONSHIPS
| Source Entity | Relationship | Target Entity | Evidence |
| :--- | :--- | :--- | :--- |
| **Figure 12.5** | `RELATED_TO` | **Encoder-Decoder Transformer Architecture** | *"Figure 12.5 | The encoder-decoder transformer architecture for machine translati"* |

#### PROVENANCE METADATA
- **Document ID**: `96`
- **Chunk ID**: `50678`
- **Page Number**: `275`
- **Filename**: `speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf`

---

### Example: Chunk 51040 — speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf (Page 348)

#### CHUNK TEXT
```text
340
CHAPTER 15
•
AUTOMATIC SPEECH RECOGNITION

We’ll then introduce two families of methods for ASR. The ﬁrst is the encoderdecoder paradigm, and we’ll introduce the baseline attention-based encoder decoder

algorithm, sometimes called Listen Attend and Spell after an early implementation.

We’ll al...
```

#### EXTRACTED ENTITIES
| Surface Mention | Canonical Name | Type | Confidence |
| :--- | :--- | :--- | :--- |
| Automatic Speech Recognition | **Automatic Speech Recognition** | `CONCEPT` | 0.95 |
| Listen Attend and Spell | **Listen Attend and Spell** | `ALGORITHM` | 0.95 |
| Whisper | **Whisper** | `MODEL` | 0.95 |
| OpenAI | **OpenAI** | `ORGANIZATION` | 0.95 |
| OWSM | **OWSM** | `MODEL` | 0.95 |
| Wav2Vec2.0 | **Wav2Vec 2.0** | `MODEL` | 0.95 |
| HuBERT | **HuBERT** | `MODEL` | 0.95 |
| self-supervised learning | **Self-Supervised Learning** | `METHOD` | 0.9 |

#### EXTRACTED RELATIONSHIPS
| Source Entity | Relationship | Target Entity | Evidence |
| :--- | :--- | :--- | :--- |
| **Whisper** | `DEVELOPED_BY` | **OpenAI** | *"OpenAI’s Whisper system"* |
| **OWSM** | `RELATED_TO` | **Whisper** | *"an open system based on the same architecture, OWSM (the Open Whisper-style Spee"* |
| **Wav2Vec 2.0** | `RELATED_TO` | **Self-Supervised Learning** | *"self-supervised speech models (sometimes called SSL for selfsupervised learning)"* |
| **HuBERT** | `RELATED_TO` | **Self-Supervised Learning** | *"self-supervised speech models (sometimes called SSL for selfsupervised learning)"* |

#### PROVENANCE METADATA
- **Document ID**: `96`
- **Chunk ID**: `51040`
- **Page Number**: `348`
- **Filename**: `speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf`

---

### Example: Chunk 51118 — speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf (Page 363)

#### CHUNK TEXT
```text
15.5
•
CTC
355

codebook that has one codeword for each of the k clusters.
codebook

Then repeat iteratively until convergence:

1. Assignment: For each vector v(i) in the dataset assign it to one of the k clusters
by choosing the one with the nearest codeword µ. Most simply we can deﬁne
‘nearest’ a...
```

#### EXTRACTED ENTITIES
| Surface Mention | Canonical Name | Type | Confidence |
| :--- | :--- | :--- | :--- |
| CTC | **Connectionist Temporal Classification** | `ALGORITHM` | 0.85 |
| codebook | **codebook** | `CONCEPT` | 0.9 |
| codeword | **codeword** | `CONCEPT` | 0.9 |
| squared Euclidean distance | **squared Euclidean distance** | `METRIC` | 0.95 |
| L2 norm | **L2 norm** | `METRIC` | 0.95 |

#### EXTRACTED RELATIONSHIPS
| Source Entity | Relationship | Target Entity | Evidence |
| :--- | :--- | :--- | :--- |
| **codebook** | `HAS_COMPONENT` | **codeword** | *"codebook that has one codeword for each of the k clusters."* |

#### PROVENANCE METADATA
- **Document ID**: `96`
- **Chunk ID**: `51118`
- **Page Number**: `363`
- **Filename**: `speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf`

---
