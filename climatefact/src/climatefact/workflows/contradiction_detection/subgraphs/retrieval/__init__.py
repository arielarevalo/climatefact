from langgraph.graph import END, START, StateGraph

from climatefact.workflows.contradiction_detection.subgraphs.retrieval.nodes.combine_and_semantic_search import (
    combine_and_semantic_search_node,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.nodes.deduplicate_results import (
    deduplicate_results_node,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.nodes.retrieve_by_ner import retrieve_by_ner_node
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.nodes.retrieve_by_regex import (
    retrieve_by_regex_node,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.nodes.retrieve_by_semantic_search import (
    retrieve_by_semantic_search_node,
)
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalConfig, RetrievalState

# Define the workflow
workflow = StateGraph(RetrievalState, config_schema=RetrievalConfig)

# Add all nodes
workflow.add_node("retrieve_by_regex", retrieve_by_regex_node)
workflow.add_node("retrieve_by_ner", retrieve_by_ner_node)
workflow.add_node("retrieve_by_semantic_search", retrieve_by_semantic_search_node)
workflow.add_node("combine_and_semantic_search", combine_and_semantic_search_node)
workflow.add_node("deduplicate_results", deduplicate_results_node)

# Set up the workflow:
# 1. All three retrieval methods run in parallel
# 2. Regex and NER feed into hybrid search
# 3. Semantic search runs independently
# 4. Both hybrid and semantic results are combined in final deduplication

workflow.add_edge(START, "retrieve_by_regex")
workflow.add_edge(START, "retrieve_by_ner")
workflow.add_edge(START, "retrieve_by_semantic_search")

workflow.add_edge("retrieve_by_regex", "combine_and_semantic_search")
workflow.add_edge("retrieve_by_ner", "combine_and_semantic_search")

workflow.add_edge("combine_and_semantic_search", "deduplicate_results")
workflow.add_edge("retrieve_by_semantic_search", "deduplicate_results")

workflow.add_edge("deduplicate_results", END)

# Compile the graph
graph = workflow.compile()
graph.name = "Hybrid Retrieval Graph"
