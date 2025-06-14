import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.concept_extractor import ConceptExtractor
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.data_loading import (
    load_concept_index,
    load_passages_from_jsonl,
    retrieve_passages_by_sentence_ids,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState

logger = logging.getLogger(__name__)


def extract_ner_concepts(text: str, extractor: ConceptExtractor) -> list[dict[str, Any]]:
    """Extract concepts using NER methods (spaCy, NLTK, entity ruler)."""
    all_concepts = extractor.extract_all_concepts(text)

    # Combine NER-based extractions
    ner_concepts = []
    ner_concepts.extend(all_concepts.get("spacy", []))
    ner_concepts.extend(all_concepts.get("nltk", []))
    ner_concepts.extend(all_concepts.get("entity_ruler", []))
    ner_concepts.extend(all_concepts.get("domain_specific", []))

    # Merge overlapping concepts
    if ner_concepts:
        ner_concepts = extractor.merge_overlapping_concepts(ner_concepts)

    return ner_concepts


def find_matching_sentence_ids(concepts: list[dict[str, Any]], concept_index: dict[str, Any]) -> list[str]:
    """Find sentence IDs that match the extracted concepts using hybrid index."""
    sentence_ids = set()
    hybrid_index = concept_index.get("hybrid_index", {})

    for concept in concepts:
        concept_text = concept["text"].lower()
        concept_type = concept["type"]

        if concept_type in hybrid_index:
            type_concepts = hybrid_index[concept_type]

            # Try exact match first
            for indexed_concept, data in type_concepts.items():
                if concept_text == indexed_concept.lower():
                    sentences = data.get("sentences", [])
                    # Extract sentence_id from each sentence object
                    sentence_ids.update([s["sentence_id"] for s in sentences])
                    logger.debug(f"Found {len(sentences)} sentences for exact match: {concept_text}")
                    break
            else:
                # Try partial matches
                for indexed_concept, data in type_concepts.items():
                    indexed_concept_lower = indexed_concept.lower()
                    if concept_text in indexed_concept_lower or indexed_concept_lower in concept_text:
                        sentences = data.get("sentences", [])
                        # Extract sentence_id from each sentence object
                        sentence_ids.update([s["sentence_id"] for s in sentences])
                        logger.debug(
                            f"Found {len(sentences)} sentences for partial match: {concept_text} -> {indexed_concept}"
                        )
                        break

    return list(sentence_ids)


def retrieve_by_ner_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Retrieves relevant passages using NER-based concept extraction.
    Extracts concepts from queries using NER methods (spaCy, NLTK, EntityRuler),
    finds matching concepts in the hybrid concept index, and retrieves corresponding passages.
    """
    queries = state.get("queries", [])
    jsonl_path = config["configurable"]["passages_jsonl_path"]
    concept_index_path = config["configurable"].get("concept_index_path")

    if not jsonl_path:
        logger.error("No 'passages_jsonl_path' provided in configuration")
        return {"ner_retrieved_data": []}

    if not concept_index_path:
        logger.error("No 'concept_index_path' provided in configuration")
        return {"ner_retrieved_data": []}

    logger.info(f"---RETRIEVING PASSAGES USING NER CONCEPTS FOR {len(queries)} QUERIES---")

    # Load passages and concept index
    all_passages = load_passages_from_jsonl(jsonl_path)
    concept_index = load_concept_index(concept_index_path)

    if not all_passages:
        logger.warning("No passages loaded, returning empty results")
        return {"ner_retrieved_data": []}

    if not concept_index:
        logger.warning("No concept index loaded, returning empty results")
        return {"ner_retrieved_data": []}

    # Initialize concept extractor (NER enabled)
    extractor = ConceptExtractor(enable_spacy=True, enable_nltk=True)

    # Log available extraction methods
    stats = extractor.get_extraction_stats()
    logger.info(
        f"NER extraction capabilities: spaCy={stats['spacy_available']}, "
        f"NLTK={stats['nltk_available']}, EntityRuler={stats['entity_ruler_available']}"
    )

    all_retrieved_passages: list[list[dict[str, Any]]] = []

    for query_text in queries:
        logger.info(f"Processing query with NER concepts: '{query_text}'")

        # Extract concepts using NER methods
        concepts = extract_ner_concepts(query_text, extractor)
        logger.info(f"Extracted {len(concepts)} NER concepts from query")

        if concepts:
            concept_types = list(set(c["type"] for c in concepts))
            extraction_sources = list(set(c["source"] for c in concepts))
            logger.debug(f"Concept types found: {concept_types}")
            logger.debug(f"Extraction sources: {extraction_sources}")

        # Find matching sentence IDs using hybrid index
        sentence_ids = find_matching_sentence_ids(concepts, concept_index)
        logger.info(f"Found {len(sentence_ids)} matching sentence IDs using hybrid index")

        # Retrieve passages by sentence IDs
        passages_for_query = retrieve_passages_by_sentence_ids(sentence_ids, all_passages)

        # Limit to top results (e.g., 15 passages max for NER as it might be more comprehensive)
        passages_for_query = passages_for_query[:15]

        all_retrieved_passages.append(passages_for_query)
        logger.info(f"Retrieved {len(passages_for_query)} passages for query '{query_text}'")

    logger.info(f"Finished NER-based retrieval for all {len(queries)} queries")

    return {"ner_retrieved_data": all_retrieved_passages}
