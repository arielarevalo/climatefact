from langgraph.graph import END, START, StateGraph

from climatefact.workflows.contradiction_detection.nodes.detect_contradictions import detect_contradictions
from climatefact.workflows.contradiction_detection.nodes.segment_sentences import segment_sentences
from climatefact.workflows.contradiction_detection.subgraphs.generation import graph as generation_graph
from climatefact.workflows.contradiction_detection.subgraphs.retrieval import graph as retrieval_graph
from climatefact.workflows.contradiction_detection.types import (
    ContradictionDetectionConfig,
    ContradictionDetectionState,
)

workflow = StateGraph(ContradictionDetectionState, config_schema=ContradictionDetectionConfig)

workflow.add_node("segment_sentences", segment_sentences)
workflow.add_node("retrieval_subgraph", retrieval_graph)
workflow.add_node("detect_contradictions", detect_contradictions)
workflow.add_node("generation_subgraph", generation_graph)

workflow.add_edge(START, "segment_sentences")
workflow.add_edge("segment_sentences", "retrieval_subgraph")
workflow.add_edge("retrieval_subgraph", "detect_contradictions")
workflow.add_edge("detect_contradictions", "generation_subgraph")
workflow.add_edge("generation_subgraph", END)

graph = workflow.compile()
graph.name = "Contradiction Detection Graph"

# Export for LangGraph CLI
__all__ = ["ContradictionDetectionConfig", "ContradictionDetectionState", "graph"]
