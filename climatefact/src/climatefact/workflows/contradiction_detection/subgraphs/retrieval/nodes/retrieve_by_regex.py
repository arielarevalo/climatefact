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


def extract_regex_concepts(text: str, extractor: ConceptExtractor) -> list[dict[str, Any]]:
    """Extract concepts using only regex patterns."""
    all_concepts = extractor.extract_all_concepts(text)
    return all_concepts.get("regex", [])


def find_matching_sentence_ids(concepts: list[dict[str, Any]], concept_index: dict[str, Any]) -> list[str]:
    """Find sentence IDs that match the extracted concepts using regex index."""
    sentence_ids = set()
    regex_index = concept_index.get("regex_index", {})

    for concept in concepts:
        concept_text = concept["text"].lower()
        concept_type = concept["type"]

        if concept_type in regex_index:
            type_concepts = regex_index[concept_type]

            # Try exact match first
            for indexed_concept, concept_data_list in type_concepts.items():
                if concept_text == indexed_concept.lower():
                    # Extract sentence IDs from the concept data list
                    for concept_data in concept_data_list:
                        sentences = concept_data.get("sentences", [])
                        sentence_ids.update(sentences)
                    logger.debug(f"Found sentences for exact match: {concept_text}")
                    break
            else:
                # Try partial matches
                for indexed_concept, concept_data_list in type_concepts.items():
                    indexed_concept_lower = indexed_concept.lower()
                    if concept_text in indexed_concept_lower or indexed_concept_lower in concept_text:
                        # Extract sentence IDs from the concept data list
                        for concept_data in concept_data_list:
                            sentences = concept_data.get("sentences", [])
                            sentence_ids.update(sentences)
                        logger.debug(f"Found sentences for partial match: {concept_text} -> {indexed_concept}")
                        break

    return list(sentence_ids)


def retrieve_by_regex_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Retrieves relevant passages using regex-based concept extraction.
    Extracts concepts from queries using regex patterns, finds matching concepts
    in the concept index, and retrieves corresponding passages.
    """
    queries = state.get("queries", [])
    jsonl_path = config["configurable"]["passages_jsonl_path"]
    concept_index_path = config["configurable"].get("concept_index_path")

    if not jsonl_path:
        logger.error("No 'passages_jsonl_path' provided in configuration")
        return {"regex_retrieved_data": []}

    if not concept_index_path:
        logger.error("No 'concept_index_path' provided in configuration")
        return {"regex_retrieved_data": []}

    logger.info(f"---RETRIEVING PASSAGES USING REGEX CONCEPTS FOR {len(queries)} QUERIES---")

    # Load passages and concept index
    all_passages = load_passages_from_jsonl(jsonl_path)
    concept_index = load_concept_index(concept_index_path)

    if not all_passages:
        logger.warning("No passages loaded, returning empty results")
        return {"regex_retrieved_data": []}

    if not concept_index:
        logger.warning("No concept index loaded, returning empty results")
        return {"regex_retrieved_data": []}

    # Initialize concept extractor (regex only)
    extractor = ConceptExtractor(enable_spacy=False, enable_nltk=False)

    all_retrieved_passages: list[list[dict[str, Any]]] = []

    for query_text in queries:
        logger.info(f"Processing query with regex concepts: '{query_text}'")

        # Extract concepts using regex
        concepts = extract_regex_concepts(query_text, extractor)
        logger.info(f"Extracted {len(concepts)} regex concepts from query")

        if concepts:
            concept_types = list(set(c["type"] for c in concepts))
            logger.debug(f"Concept types found: {concept_types}")

        # Find matching sentence IDs
        sentence_ids = find_matching_sentence_ids(concepts, concept_index)
        logger.info(f"Found {len(sentence_ids)} matching sentence IDs")

        # Retrieve passages by sentence IDs
        passages_for_query = retrieve_passages_by_sentence_ids(sentence_ids, all_passages)

        # Limit to top results (e.g., 10 passages max)
        passages_for_query = passages_for_query[:10]

        all_retrieved_passages.append(passages_for_query)
        logger.info(f"Retrieved {len(passages_for_query)} passages for query '{query_text}'")

    logger.info(f"Finished regex-based retrieval for all {len(queries)} queries")

    return {"regex_retrieved_data": all_retrieved_passages}
