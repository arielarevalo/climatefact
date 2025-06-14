import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.data_loading import deduplicate_passages
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState

logger = logging.getLogger(__name__)


def deduplicate_results_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Final deduplication step that combines results from semantic search and
    hybrid (regex+NER+semantic) retrieval methods.
    """
    queries = state.get("queries", [])
    semantic_data = state.get("semantic_retrieved_data", []) or []
    hybrid_data = state.get("hybrid_retrieved_data", []) or []

    logger.info(f"---FINAL DEDUPLICATION FOR {len(queries)} QUERIES---")

    final_retrieved_data: list[list[dict[str, Any]]] = []

    for i, query in enumerate(queries):
        logger.info(f"Final deduplication for query {i}: '{query}'")

        # Get passages from both sources
        semantic_passages = semantic_data[i] if i < len(semantic_data) else []
        hybrid_passages = hybrid_data[i] if i < len(hybrid_data) else []

        logger.info(f"Semantic passages: {len(semantic_passages)}, Hybrid passages: {len(hybrid_passages)}")

        # Combine all passages and deduplicate
        all_passages = semantic_passages + hybrid_passages
        deduplicated_passages = deduplicate_passages(all_passages)

        # Limit final results (e.g., top 10 passages)
        final_passages = deduplicated_passages[:10]

        final_retrieved_data.append(final_passages)

        logger.info(f"Final count for query '{query}': {len(final_passages)} passages (from {len(all_passages)} total)")

    logger.info(f"Finished final deduplication for all {len(queries)} queries")

    return {"retrieved_data_for_queries": final_retrieved_data}
