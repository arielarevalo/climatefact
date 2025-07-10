from typing import List, Dict, TypedDict, Any

class ContradictionEvidence(TypedDict):
    contradictory_passage: str
    source: str

class ContradictionResult(TypedDict):
    sentence: str
    contradictions: List[ContradictionEvidence]
    has_contradictions: bool

class GenerationState(TypedDict):
    """
    State for the generation subgraph.
    """
    contradiction_results: List[ContradictionResult]
    report: str
