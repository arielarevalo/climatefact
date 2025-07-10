from typing import List, Dict, TypedDict, Any, Optional, Annotated

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
    queries: List[str]
    regex_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    ner_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    semantic_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    hybrid_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    retrieved_data_for_queries: Optional[List[List[Dict[str, Any]]]]
