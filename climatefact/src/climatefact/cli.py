#!/usr/bin/env python3
"""
CLI entrypoint for ClimateFact contradiction detection.
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection import graph


def main():
    """Main CLI entrypoint for contradiction detection."""
    parser = argparse.ArgumentParser(
        description="ClimateFact - Detect contradictions in climate statements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("statement", help="Climate statement to analyze for contradictions")

    parser.add_argument("--passages-path", default="../data/passages.jsonl", help="Path to the passages JSONL file")

    parser.add_argument(
        "--concept-index-path", default="../data/concept_index.json", help="Path to the concept index JSON file"
    )

    parser.add_argument(
        "--hybrid-retrieval-top-k", type=int, default=5, help="Number of top passages to retrieve for hybrid retrieval"
    )

    parser.add_argument(
        "--semantic-search-top-k", type=int, default=5, help="Number of top passages to retrieve for semantic search"
    )

    parser.add_argument("--output-format", choices=["json", "text"], default="text", help="Output format for results")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output with raw results")

    args = parser.parse_args()

    # Validate input
    if not args.statement.strip():
        print("Error: Statement cannot be empty", file=sys.stderr)
        sys.exit(1)

    # Check if paths exist
    passages_path = Path(args.passages_path)
    concept_index_path = Path(args.concept_index_path)

    if not passages_path.exists():
        print(f"Error: Passages file not found: {passages_path}", file=sys.stderr)
        sys.exit(1)

    if not concept_index_path.exists():
        print(f"Error: Concept index file not found: {concept_index_path}", file=sys.stderr)
        sys.exit(1)

    # Configuration for the graph
    config = RunnableConfig(
        configurable={
            "passages_jsonl_path": str(passages_path),
            "concept_index_path": str(concept_index_path),
            "hybrid_retrieval_top_k": args.hybrid_retrieval_top_k,
            "semantic_search_top_k": args.semantic_search_top_k,
        }
    )

    # Initial state
    initial_state = {
        "input_text": args.statement.strip(),
        "queries": None,
        "retrieved_data_for_queries": None,
        "contradiction_results": None,
        "report": None,
    }

    print("🔄 Analyzing statement for contradictions...")
    print(f"Statement: {args.statement}")
    print("-" * 50)

    try:
        # Run the graph
        result = graph.invoke(initial_state, config=config)

        if args.output_format == "json":
            # Output as JSON
            print(json.dumps(result, indent=2, default=str))
        else:
            # Output as formatted text
            print("✅ Analysis complete!")
            print()

            if result.get("report"):
                print("📋 REPORT:")
                print(result["report"])
                print()
            else:
                print("⚠️ No report generated.")
                print()

            # Show contradiction results if available
            if result.get("contradiction_results"):
                print("🔍 CONTRADICTION RESULTS:")
                for i, contradiction_result in enumerate(result["contradiction_results"], 1):
                    sentence = contradiction_result.get("sentence", "")
                    has_contradictions = contradiction_result.get("has_contradictions", False)
                    contradictions = contradiction_result.get("contradictions", [])

                    print(f"{i}. Sentence: {sentence}")
                    print(f"   Has contradictions: {has_contradictions}")

                    if contradictions:
                        print("   Contradictory evidence:")
                        for j, evidence in enumerate(contradictions, 1):
                            passage = evidence.get("contradictory_passage", "")
                            source = evidence.get("source", "Unknown")
                            print(f"     {j}. Source: {source}")
                            print(f"        Passage: {passage[:200]}{'...' if len(passage) > 200 else ''}")
                    print()

            # Show verbose output if requested
            if args.verbose:
                print("🔧 RAW RESULTS:")
                print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"❌ Error occurred during analysis: {e!s}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
