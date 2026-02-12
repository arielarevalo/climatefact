from typing import Any, TypedDict


class RetrievalConfig(TypedDict):
    """Configuration for the retrieval subgraph."""

    passages_jsonl_path: str
    concept_index_path: str
    hybrid_retrieval_top_k: int
    semantic_search_top_k: int


class RetrievalState(TypedDict):
    """
    State for the retrieval graph.
    It processes a list of queries and retrieves passages from a JSONL file.
    This is a subset of ContradictionDetectionState.
    """

    queries: list[str]
    regex_retrieved_data: list[list[dict[str, Any]]] | None
    ner_retrieved_data: list[list[dict[str, Any]]] | None
    semantic_retrieved_data: list[list[dict[str, Any]]] | None
    hybrid_retrieved_data: list[list[dict[str, Any]]] | None
    retrieved_data_for_queries: list[list[dict[str, Any]]] | None
