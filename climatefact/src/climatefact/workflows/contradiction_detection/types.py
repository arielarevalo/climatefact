from typing import List, Dict, TypedDict, Any, Optional, Annotated
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalConfig

class ContradictionDetectionConfig(RetrievalConfig):
    """Configuration for the contradiction detection workflow."""
    pass

class ContradictionEvidence(TypedDict):
    contradictory_passage: str
    source: str

class ContradictionResult(TypedDict):
    sentence: str
    contradictions: List[ContradictionEvidence]
    has_contradictions: bool

class ContradictionDetectionState(TypedDict):
    """
    State for the main contradiction detection workflow.
    Contains RetrievalState as a subset plus additional fields.
    """
    input_text: str
    queries: List[str]
    regex_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    ner_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    semantic_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    hybrid_retrieved_data: Optional[List[List[Dict[str, Any]]]]
    retrieved_data_for_queries: Optional[List[List[Dict[str, Any]]]]
    contradiction_results: Optional[List[ContradictionResult]]
    report: Optional[str]
