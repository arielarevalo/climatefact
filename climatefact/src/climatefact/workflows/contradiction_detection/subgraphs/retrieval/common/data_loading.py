"""Shared utilities for loading and retrieving passages."""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_passages_from_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Load passages from a JSONL file."""
    passages = []
    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL file not found at {jsonl_path}")
        return passages

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                passage = json.loads(line.strip())
                passages.append(passage)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    logger.info(f"Loaded {len(passages)} passages from {jsonl_path}")
    return passages


def load_concept_index(index_path: str) -> dict[str, Any]:
    """Load the concept index from JSON file."""
    if not os.path.exists(index_path):
        logger.warning(f"Concept index not found at {index_path}")
        return {}

    try:
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load concept index from {index_path}: {e}")
        return {}


def retrieve_passages_by_sentence_ids(sentence_ids: list[str], passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieve passages that match the given sentence IDs."""
    if not sentence_ids:
        return []

    sentence_id_set = set(sentence_ids)
    matching_passages = []

    for passage in passages:
        passage_id = passage.get("id", "")
        if passage_id in sentence_id_set:
            # Keep embeddings for hybrid search - they'll be cleaned up later
            matching_passages.append(passage)

    return matching_passages


def deduplicate_passages(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate passages based on their ID, preserving order."""
    seen_ids = set()
    deduplicated = []

    for passage in passages:
        passage_id = passage.get("id", "")
        if passage_id and passage_id not in seen_ids:
            seen_ids.add(passage_id)
            deduplicated.append(passage)
        elif not passage_id:
            # If no ID, check text similarity to avoid duplicates
            passage_text = passage.get("text", "")
            if passage_text and not any(p.get("text", "") == passage_text for p in deduplicated):
                deduplicated.append(passage)

    return deduplicated
