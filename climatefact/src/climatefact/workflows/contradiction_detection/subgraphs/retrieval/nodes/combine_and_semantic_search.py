import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.data_loading import deduplicate_passages
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.common.embeddings import vector_search
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState

logger = logging.getLogger(__name__)


def combine_and_semantic_search_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Combines passages from regex and NER retrieval, applies semantic search.
    Results are stored in hybrid_retrieved_data.
    """
    queries = state.get("queries", [])
    regex_data = state.get("regex_retrieved_data", [])
    ner_data = state.get("ner_retrieved_data", [])

    logger.info(f"---COMBINING REGEX+NER AND APPLYING SEMANTIC SEARCH FOR {len(queries)} QUERIES---")

    hybrid_retrieved_data: list[list[dict[str, Any]]] = []

    top_k = config["configurable"].get("hybrid_retrieval_top_k", 5)
    logger.info(f"Using top_k={top_k} for hybrid retrieval")
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i}: '{query}'")

        # Get passages from both methods for this query
        regex_passages = regex_data[i] if i < len(regex_data) else []
        ner_passages = ner_data[i] if i < len(ner_data) else []

        logger.info(f"Regex passages: {len(regex_passages)}, NER passages: {len(ner_passages)}")

        # Combine and deduplicate passages
        combined_passages = deduplicate_passages(regex_passages + ner_passages)
        logger.info(f"Combined unique passages: {len(combined_passages)}")

        if combined_passages:
            # Apply semantic search on combined passages
            top_passages = vector_search(query, combined_passages, top_k=3)
            logger.info(f"Top semantic matches: {len(top_passages)}")

            # Clean up passages (remove embeddings)
            cleaned_passages = []
            for passage in top_passages:
                cleaned_passage = passage.copy()
                if "embedding" in cleaned_passage:
                    del cleaned_passage["embedding"]
                cleaned_passages.append(cleaned_passage)

            hybrid_retrieved_data.append(cleaned_passages)
        else:
            logger.info("No passages to process for this query")
            hybrid_retrieved_data.append([])

    logger.info(f"Finished hybrid retrieval for all {len(queries)} queries")

    return {"hybrid_retrieved_data": hybrid_retrieved_data}
