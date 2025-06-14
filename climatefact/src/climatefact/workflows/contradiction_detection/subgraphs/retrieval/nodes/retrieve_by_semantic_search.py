import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.data_loading import (
    load_passages_from_jsonl,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.embeddings import vector_search
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState

logger = logging.getLogger(__name__)


def retrieve_by_semantic_search_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Retrieves relevant passages for a list of given queries from a JSONL file.
    Uses vector similarity if embeddings are available, falls back to text search.
    Gets 'passages_jsonl_path' from configuration.
    """
    queries = state.get("queries", [])
    jsonl_path = config["configurable"]["passages_jsonl_path"]

    if not jsonl_path:
        logger.error("No 'passages_jsonl_path' provided in configuration")
        return {"semantic_retrieved_data": []}

    logger.info(f"---RETRIEVING PASSAGES FOR {len(queries)} QUERIES FROM {jsonl_path}---")

    # Load all passages from JSONL file
    all_passages = load_passages_from_jsonl(jsonl_path)
    if not all_passages:
        logger.warning("No passages loaded, returning empty results")
        return {"semantic_retrieved_data": []}

    all_retrieved_passages: list[list[dict[str, Any]]] = []

    top_k = config["configurable"].get("semantic_search_top_k", 5)
    logger.info(f"Using top_k={top_k} for semantic retrieval")
    for query_text in queries:
        logger.info(f"Retrieving for query: '{query_text}'")

        logger.info("Using vector search with embeddings")
        passages_for_query = vector_search(query_text, all_passages, top_k=top_k)

        # Clean up passages for output (remove embeddings to save memory)
        cleaned_passages = []
        for passage in passages_for_query:
            cleaned_passage = passage.copy()
            if "embedding" in cleaned_passage:
                del cleaned_passage["embedding"]  # Remove large embedding vector
            cleaned_passages.append(cleaned_passage)

        all_retrieved_passages.append(cleaned_passages)
        logger.info(f"Retrieved {len(passages_for_query)} passages for query '{query_text}'.")

    # Add detailed logging to verify state content
    logger.info(f"Finished semantic retrieval for all {len(queries)} queries.")
    logger.info(f"Total retrieved passage sets: {len(all_retrieved_passages)}")
    for i, passage_set in enumerate(all_retrieved_passages):
        logger.info(f"Query {i}: {len(passage_set)} passages retrieved")
        for j, passage in enumerate(passage_set):
            logger.debug(f"  Passage {j}: {passage.get('text', '')[:100]}...")

    return {"semantic_retrieved_data": all_retrieved_passages}
