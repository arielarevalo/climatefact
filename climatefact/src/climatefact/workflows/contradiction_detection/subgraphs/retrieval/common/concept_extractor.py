#!/usr/bin/env python3
"""
Concept Extractor Module

This module provides a standalone concept extraction system that can be used
to extract concepts from arbitrary text using the same hybrid approach as the
main build_concept_index.py script.

Usage:
    from concept_extractor import ConceptExtractor

    extractor = ConceptExtractor()
    concepts = extractor.extract_concepts("Your text here")
"""

import importlib.util
import json
import logging
import re
from typing import Any

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.config import (
    CONFIDENCE_THRESHOLDS,
    DOMAIN_SPECIFIC_PATTERNS,
    ENTITY_RULER_PATTERNS,
    EXTRACTION_PRIORITY,
    NLTK_STOPWORDS_LANG,
    PATTERN_SCHEMA,
    REGEX_MAP,
    SPACY_MODEL,
)

# Optional NLP libraries with graceful fallback

SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None

try:
    import nltk

    NLTK_AVAILABLE = True
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)
    nltk.download("stopwords", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class ConceptExtractor:
    """Hybrid concept extractor combining regex, NER, and EntityRuler approaches"""

    def __init__(self, enable_spacy: bool = True, enable_nltk: bool = True):
        """
        Initialize the concept extractor.

        Args:
            enable_spacy: Whether to use spaCy NER (default: True)
            enable_nltk: Whether to use NLTK NER (default: True)
        """
        self.nlp = None
        self.stopwords = set()
        self.entity_ruler = None
        self.enable_spacy = enable_spacy and SPACY_AVAILABLE
        self.enable_nltk = enable_nltk and NLTK_AVAILABLE
        self._setup_models()

    def _setup_models(self):
        """Initialize NLP models and resources"""
        if self.enable_spacy and SPACY_AVAILABLE:
            try:
                import spacy as spacy_module

                self.nlp = spacy_module.load(SPACY_MODEL)
                logger.info(f"Successfully loaded spaCy {SPACY_MODEL} model")
                self._setup_entity_ruler()
            except OSError:
                logger.warning(f"spaCy {SPACY_MODEL} model not found. spaCy extraction disabled.")
                self.nlp = None
                self.enable_spacy = False

        if self.enable_nltk:
            try:
                import nltk as nltk_module
                from nltk.corpus import stopwords as nltk_stopwords

                nltk_module.download("punkt", quiet=True)
                nltk_module.download("averaged_perceptron_tagger", quiet=True)
                nltk_module.download("maxent_ne_chunker", quiet=True)
                nltk_module.download("words", quiet=True)
                nltk_module.download("stopwords", quiet=True)
                self.stopwords = set(nltk_stopwords.words(NLTK_STOPWORDS_LANG))
                logger.info("Successfully initialized NLTK resources")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK resources: {e}")
                self.enable_nltk = False

    def _setup_entity_ruler(self):
        """Setup EntityRuler with climate science patterns"""
        if not self.nlp:
            return

        try:
            # Add EntityRuler to the pipeline
            if "entity_ruler" not in self.nlp.pipe_names:
                self.entity_ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                self.entity_ruler = self.nlp.get_pipe("entity_ruler")

            # Create EntityRuler patterns
            entity_patterns = []

            for concept_key, variations in ENTITY_RULER_PATTERNS.items():
                concept_type = PATTERN_SCHEMA.get(concept_key, "UNKNOWN")
                for variation in variations:
                    entity_patterns.append({"label": concept_type, "pattern": variation, "id": concept_key})

            # Add patterns to EntityRuler
            if entity_patterns and hasattr(self.entity_ruler, "add_patterns"):
                self.entity_ruler.add_patterns(entity_patterns)  # type: ignore
                logger.info(f"Added {len(entity_patterns)} patterns to EntityRuler")

        except Exception as e:
            logger.warning(f"Failed to setup EntityRuler: {e}")
            self.entity_ruler = None

    def extract_spacy_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using spaCy NER"""
        if not self.nlp or not self.enable_spacy:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Map spaCy entity types to our schema
            entity_type = self._map_spacy_entity_type(ent.label_)
            if entity_type:
                entities.append(
                    {
                        "text": ent.text,
                        "type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": CONFIDENCE_THRESHOLDS["spacy"],
                        "source": "spacy",
                    }
                )

        return entities

    def extract_nltk_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using NLTK NER"""
        if not self.enable_nltk:
            return []

        try:
            # Import NLTK functions locally
            from nltk.chunk import ne_chunk
            from nltk.tag import pos_tag
            from nltk.tokenize import word_tokenize

            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)

            entities = []
            current_entity = []
            current_label = None

            for chunk in chunks:
                if hasattr(chunk, "label"):
                    # This is a named entity
                    if current_label != chunk.label():
                        # Flush previous entity
                        if current_entity:
                            entity_text = " ".join(current_entity)
                            entity_type = self._map_nltk_entity_type(current_label)
                            if entity_type:
                                # Find the entity position in text (approximate)
                                start_pos = text.lower().find(entity_text.lower())
                                end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                                entities.append(
                                    {
                                        "text": entity_text,
                                        "type": entity_type,
                                        "start": max(0, start_pos),
                                        "end": max(len(entity_text), end_pos),
                                        "confidence": CONFIDENCE_THRESHOLDS["nltk"],
                                        "source": "nltk",
                                    }
                                )

                        # Start new entity
                        current_entity = [chunk[0][0]]
                        current_label = chunk.label()
                    else:
                        # Continue current entity
                        current_entity.append(chunk[0][0])
                else:
                    # Flush current entity if any
                    if current_entity:
                        entity_text = " ".join(current_entity)
                        entity_type = self._map_nltk_entity_type(current_label)
                        if entity_type:
                            # Find the entity position in text (approximate)
                            start_pos = text.lower().find(entity_text.lower())
                            end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                            entities.append(
                                {
                                    "text": entity_text,
                                    "type": entity_type,
                                    "start": max(0, start_pos),
                                    "end": max(len(entity_text), end_pos),
                                    "confidence": CONFIDENCE_THRESHOLDS["nltk"],
                                    "source": "nltk",
                                }
                            )
                        current_entity = []
                        current_label = None

            # Handle final entity
            if current_entity:
                entity_text = " ".join(current_entity)
                entity_type = self._map_nltk_entity_type(current_label)
                if entity_type:
                    # Find the entity position in text (approximate)
                    start_pos = text.lower().find(entity_text.lower())
                    end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
                    entities.append(
                        {
                            "text": entity_text,
                            "type": entity_type,
                            "start": max(0, start_pos),
                            "end": max(len(entity_text), end_pos),
                            "confidence": CONFIDENCE_THRESHOLDS["nltk"],
                            "source": "nltk",
                        }
                    )

            return entities

        except Exception as e:
            logger.error(f"Error in NLTK entity extraction: {e}")
            return []

    def extract_entity_ruler_concepts(self, text: str) -> list[dict[str, Any]]:
        """Extract concepts using spaCy EntityRuler"""
        if not self.nlp or not self.entity_ruler or not self.enable_spacy:
            return []

        doc = self.nlp(text)
        concepts = []

        for ent in doc.ents:
            # Check if this entity was matched by EntityRuler
            if ent.ent_id_:  # EntityRuler entities have ent_id_
                concepts.append(
                    {
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": CONFIDENCE_THRESHOLDS["entity_ruler"],
                        "source": "entity_ruler",
                        "ruler_id": ent.ent_id_,
                    }
                )

        return concepts

    def _map_spacy_entity_type(self, spacy_label: str) -> str | None:
        """Map spaCy entity labels to our concept schema"""
        mapping = {
            "ORG": "AGENCIES",
            "GPE": "LOCATION",
            "PERSON": "PERSON",
            "DATE": "TEMPORAL",
            "MONEY": "ECONOMIC",
            "PERCENT": "QUANTITATIVE",
            "QUANTITY": "QUANTITATIVE",
            "CARDINAL": "QUANTITATIVE",
            "PRODUCT": "TECHNOLOGY",
            "EVENT": "CLIMATE_VAR",
            "FAC": "TECHNOLOGY",
            "LANGUAGE": "OTHER",
            "LAW": "POLICY",
            "LOC": "LOCATION",
            "NORP": "SOCIAL",
            "ORDINAL": "QUANTITATIVE",
            "TIME": "TEMPORAL",
            "WORK_OF_ART": "REPORTS",
        }
        return mapping.get(spacy_label)

    def _map_nltk_entity_type(self, nltk_label: str | None) -> str | None:
        """Map NLTK entity labels to our concept schema"""
        if not nltk_label:
            return None
        mapping = {
            "ORGANIZATION": "AGENCIES",
            "PERSON": "PERSON",
            "LOCATION": "LOCATION",
            "GPE": "LOCATION",
            "GSP": "LOCATION",
            "FACILITY": "TECHNOLOGY",
        }
        return mapping.get(nltk_label)

    def extract_domain_specific_concepts(self, text: str) -> list[dict[str, Any]]:
        """Extract climate science specific concepts using custom rules"""
        concepts = []

        for concept_type, patterns in DOMAIN_SPECIFIC_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    concepts.append(
                        {
                            "text": match.group(),
                            "type": concept_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": CONFIDENCE_THRESHOLDS["domain_specific"],
                            "source": "domain_specific",
                        }
                    )

        return concepts

    def extract_regex_concepts(self, text: str) -> list[dict[str, Any]]:
        """Extract concepts using the existing regex patterns"""
        concepts = []

        for key, pattern in REGEX_MAP.items():
            label = PATTERN_SCHEMA.get(key, "UNKNOWN")

            for match in pattern.finditer(text):
                concepts.append(
                    {
                        "text": match.group(),
                        "type": label,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": CONFIDENCE_THRESHOLDS["regex"],
                        "source": "regex",
                        "regex_key": key,
                    }
                )

        return concepts

    def extract_all_concepts(self, text: str) -> dict[str, list[dict[str, Any]]]:
        """Extract concepts using all available methods"""
        results = {"regex": [], "entity_ruler": [], "spacy": [], "nltk": [], "domain_specific": []}

        # Extract using EntityRuler (prioritized over regex for exact matches)
        results["entity_ruler"] = self.extract_entity_ruler_concepts(text)

        # Extract using existing regex patterns (for complex patterns not suitable for EntityRuler)
        results["regex"] = self.extract_regex_concepts(text)

        # Extract using various NER approaches
        results["spacy"] = self.extract_spacy_entities(text)
        results["nltk"] = self.extract_nltk_entities(text)
        results["domain_specific"] = self.extract_domain_specific_concepts(text)

        return results

    def merge_overlapping_concepts(self, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge overlapping concepts, prioritizing higher confidence and longer spans"""
        if not concepts:
            return []

        # Filter out concepts without required fields and sort by start position, then by confidence (descending)
        valid_concepts = [c for c in concepts if "start" in c and "end" in c and "confidence" in c]
        sorted_concepts = sorted(valid_concepts, key=lambda x: (x["start"], -x["confidence"]))

        merged = []
        for concept in sorted_concepts:
            # Check for overlap with existing concepts
            overlapping = False
            for i, existing in enumerate(merged):
                if self._concepts_overlap(concept, existing):
                    # Choose the better concept
                    if self._should_replace_concept(concept, existing):
                        merged[i] = concept
                    overlapping = True
                    break

            if not overlapping:
                merged.append(concept)

        return merged

    def _concepts_overlap(self, concept1: dict[str, Any], concept2: dict[str, Any]) -> bool:
        """Check if two concepts overlap in text span"""
        return not (concept1["end"] <= concept2["start"] or concept2["end"] <= concept1["start"])

    def _should_replace_concept(self, new_concept: dict[str, Any], existing_concept: dict[str, Any]) -> bool:
        """Determine if new concept should replace existing one"""
        new_priority = EXTRACTION_PRIORITY.get(new_concept["source"], 0)
        existing_priority = EXTRACTION_PRIORITY.get(existing_concept["source"], 0)

        # Prioritize by source first
        if new_priority != existing_priority:
            return new_priority > existing_priority

        # If same source, prioritize longer spans
        new_length = new_concept["end"] - new_concept["start"]
        existing_length = existing_concept["end"] - existing_concept["start"]
        if new_length != existing_length:
            return new_length > existing_length

        # Finally, prioritize higher confidence
        return new_concept["confidence"] > existing_concept["confidence"]

    def extract_concepts(self, text: str, merge_overlapping: bool = True) -> list[dict[str, Any]]:
        """
        Extract concepts from text using all available methods.

        Args:
            text: Input text to extract concepts from
            merge_overlapping: Whether to merge overlapping concepts (default: True)

        Returns:
            List of concept dictionaries with keys: text, type, start, end, confidence, source
        """
        # Extract concepts using all methods
        all_concepts = self.extract_all_concepts(text)

        # Flatten concepts
        flat_concepts = []
        for concepts in all_concepts.values():
            flat_concepts.extend(concepts)

        # Merge overlapping concepts if requested
        if merge_overlapping:
            return self.merge_overlapping_concepts(flat_concepts)
        else:
            return flat_concepts

    def get_extraction_stats(self) -> dict[str, Any]:
        """Get information about available extraction methods"""
        return {
            "spacy_available": self.enable_spacy,
            "nltk_available": self.enable_nltk,
            "entity_ruler_available": self.entity_ruler is not None,
            "regex_patterns_count": len(REGEX_MAP),
            "domain_patterns_count": len(DOMAIN_SPECIFIC_PATTERNS),
            "entity_ruler_patterns_count": len(ENTITY_RULER_PATTERNS) if ENTITY_RULER_PATTERNS else 0,
        }


class ConceptIndexQuerier:
    """Query concepts from a pre-built concept index"""

    def __init__(self, index_path: str):
        """
        Initialize the querier with a concept index file.

        Args:
            index_path: Path to the concept index JSON file
        """
        self.index_path = index_path
        self.index = self._load_index()
        self.extractor = ConceptExtractor()

    def _load_index(self) -> dict[str, Any]:
        """Load the concept index from file"""
        try:
            with open(self.index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load concept index from {self.index_path}: {e}")
            return {}

    def query_text(self, text: str, use_hybrid: bool = True) -> dict[str, Any]:
        """
        Query the concept index with arbitrary text.

        Args:
            text: Input text to extract concepts from and query against index
            use_hybrid: Whether to use hybrid index (True) or regex index (False)

        Returns:
            Dictionary with extracted concepts and matching sentences from index
        """
        # Extract concepts from the input text
        extracted_concepts = self.extractor.extract_concepts(text)

        # Query the index for matching concepts
        matches = {}
        index_to_use = self.index.get("hybrid_index" if use_hybrid else "regex_index", {})

        for concept in extracted_concepts:
            concept_text = concept["text"].lower()
            concept_type = concept["type"]

            # Look for exact matches in the index
            if concept_type in index_to_use:
                type_concepts = index_to_use[concept_type]

                # Try case-insensitive exact match first
                exact_match_key = None
                for indexed_concept in type_concepts:
                    if concept_text == indexed_concept.lower():
                        exact_match_key = indexed_concept
                        break

                if exact_match_key:
                    matches[concept_text] = {
                        "concept_info": concept,
                        "index_data": type_concepts[exact_match_key],
                        "match_type": "exact",
                    }
                else:
                    # Try case-insensitive partial matches
                    for indexed_concept, data in type_concepts.items():
                        indexed_concept_lower = indexed_concept.lower()
                        if concept_text in indexed_concept_lower or indexed_concept_lower in concept_text:
                            matches[f"{concept_text} → {indexed_concept}"] = {
                                "concept_info": concept,
                                "index_data": data,
                                "match_type": "partial",
                            }
                            break

        return {
            "input_text": text,
            "extracted_concepts": extracted_concepts,
            "index_matches": matches,
            "stats": {
                "total_extracted": len(extracted_concepts),
                "total_matches": len(matches),
                "extraction_methods": list(set(c["source"] for c in extracted_concepts)),
                "concept_types": list(set(c["type"] for c in extracted_concepts)),
            },
        }

    def get_concept_sentences(
        self, concept_text: str, concept_type: str | None = None, use_hybrid: bool = True
    ) -> list[str]:
        """
        Get sentence IDs that contain a specific concept.

        Args:
            concept_text: The concept text to search for
            concept_type: Optional concept type to narrow search
            use_hybrid: Whether to use hybrid index (True) or regex index (False)

        Returns:
            List of sentence IDs containing the concept
        """
        index_to_use = self.index.get("hybrid_index" if use_hybrid else "regex_index", {})
        concept_text_lower = concept_text.lower()
        sentence_ids = []

        types_to_search = [concept_type] if concept_type else index_to_use.keys()

        for ctype in types_to_search:
            if ctype in index_to_use:
                for indexed_concept, data in index_to_use[ctype].items():
                    if concept_text_lower == indexed_concept.lower():
                        if use_hybrid:
                            sentence_ids.extend([s["sentence_id"] for s in data.get("sentences", [])])
                        else:
                            sentence_ids.extend(data.get("sentences", []))
                        break

        return list(set(sentence_ids))  # Remove duplicates
