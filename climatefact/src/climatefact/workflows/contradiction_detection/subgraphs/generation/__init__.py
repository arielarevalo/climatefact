from langgraph.graph import StateGraph, START, END
from climatefact.workflows.contradiction_detection.subgraphs.generation.nodes.generate_report import generate_report
from climatefact.workflows.contradiction_detection.subgraphs.generation.types import GenerationState

workflow = StateGraph(GenerationState)

workflow.add_node("generate_report", generate_report)

workflow.add_edge(START, "generate_report")
workflow.add_edge("generate_report", END)

graph = workflow.compile()
graph.name = "Generation Graph"
