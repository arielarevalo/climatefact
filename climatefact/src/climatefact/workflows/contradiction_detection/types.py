from typing import Any, TypedDict

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalConfig


class ContradictionDetectionConfig(RetrievalConfig):
    """Configuration for the contradiction detection workflow."""

    pass


class ContradictionEvidence(TypedDict):
    contradictory_passage: str
    source: str


class ContradictionResult(TypedDict):
    sentence: str
    contradictions: list[ContradictionEvidence]
    has_contradictions: bool


class ContradictionDetectionState(TypedDict):
    """
    State for the main contradiction detection workflow.
    Contains RetrievalState as a subset plus additional fields.
    """

    input_text: str
    queries: list[str]
    regex_retrieved_data: list[list[dict[str, Any]]] | None
    ner_retrieved_data: list[list[dict[str, Any]]] | None
    semantic_retrieved_data: list[list[dict[str, Any]]] | None
    hybrid_retrieved_data: list[list[dict[str, Any]]] | None
    retrieved_data_for_queries: list[list[dict[str, Any]]] | None
    contradiction_results: list[ContradictionResult] | None
    report: str | None
