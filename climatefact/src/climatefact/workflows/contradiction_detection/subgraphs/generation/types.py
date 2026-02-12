from typing import TypedDict

from climatefact.workflows.contradiction_detection.types import ContradictionResult


class GenerationState(TypedDict):
    """
    State for the generation subgraph.
    """

    contradiction_results: list[ContradictionResult]
    report: str
