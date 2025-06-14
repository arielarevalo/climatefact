"""
Example usage of the concurrent contradiction detection implementation.

This example demonstrates how the new concurrent NLI processing works compared to
the original serial implementation.
"""

from typing import cast

from climatefact.workflows.contradiction_detection.types import ContradictionDetectionState


def create_sample_state() -> ContradictionDetectionState:
    """Create a sample state for demonstration."""
    return cast(
        ContradictionDetectionState,
        {
            "input_text": "Example climate statements for testing",
            "queries": [
                "Climate change is not happening",
                "Global temperatures are decreasing",
                "Sea levels are falling worldwide",
            ],
            "regex_retrieved_data": None,
            "ner_retrieved_data": None,
            "semantic_retrieved_data": None,
            "hybrid_retrieved_data": None,
            "retrieved_data_for_queries": [
                [  # Passages for first query
                    {
                        "text": "Climate change is extensively documented by scientific research",
                        "source": "IPCC Report 2023",
                    },
                    {"text": "Temperature records show consistent warming trends", "source": "NASA Climate Data"},
                    {"text": "Greenhouse gas concentrations continue to rise", "source": "NOAA Atmospheric Data"},
                ],
                [  # Passages for second query
                    {
                        "text": "Global mean temperature has increased by 1.1°C since pre-industrial times",
                        "source": "WMO State of Climate 2023",
                    },
                    {"text": "The last decade was the warmest on record", "source": "Berkeley Earth Temperature Study"},
                ],
                [  # Passages for third query
                    {
                        "text": "Sea levels have risen approximately 20cm since 1900",
                        "source": "Coastal Research Institute",
                    },
                    {
                        "text": "Ice sheet melting contributes to accelerating sea level rise",
                        "source": "Glaciology Research Center",
                    },
                    {
                        "text": "Thermal expansion of oceans drives sea level increase",
                        "source": "Oceanographic Institute",
                    },
                ],
            ],
            "contradiction_results": None,
            "report": None,
        },
    )


def demonstrate_concurrent_processing():
    """Demonstrate the concurrent contradiction detection."""
    print("Concurrent Contradiction Detection Example")
    print("=" * 50)

    # Create sample state
    state = create_sample_state()

    print(f"Input queries: {len(state['queries'])}")
    for i, query in enumerate(state["queries"]):
        passages = state["retrieved_data_for_queries"][i] if state["retrieved_data_for_queries"] else []
        print(f"  {i + 1}. '{query}' ({len(passages)} passages)")

    print(
        "\nTotal NLI comparisons to perform: "
        f"{sum(len(passages) for passages in (state['retrieved_data_for_queries'] or []))}"
    )

    print("\n" + "-" * 50)
    print("Key benefits of the concurrent implementation:")
    print("1. ✓ All NLI calls are processed concurrently using ThreadPoolExecutor")
    print("2. ✓ Resource initialization (NLI client) happens once, outside the concurrent calls")
    print("3. ✓ Proper separation of concerns:")
    print("   - Configuration and client setup")
    print("   - Individual NLI model calls")
    print("   - Concurrency orchestration")
    print("   - Result organization")
    print("4. ✓ Robust error handling for individual tasks")
    print("5. ✓ Configurable concurrency level (max_workers)")
    print("6. ✓ Comprehensive test coverage")

    print("\n" + "-" * 50)
    print("Architecture layers:")
    print("1. detect_contradictions() - Main entry point")
    print("2. _prepare_nli_tasks() - Task preparation")
    print("3. _process_nli_tasks_concurrently() - Concurrency orchestration")
    print("4. _process_nli_task() - Single task processor")
    print("5. _call_nli_model_sync() - Actual NLI model call")
    print("6. _organize_nli_results() - Result aggregation")

    # Note: We don't actually call detect_contradictions here because we'd need
    # real Azure ML NLI credentials, but the structure is ready to use
    print("\n" + "-" * 50)
    print("To run with real NLI model, set environment variables:")
    print("- AZURE_INFERENCE_ENDPOINT")
    print("- AZURE_INFERENCE_CREDENTIAL")
    print("Then call: detect_contradictions(state)")


if __name__ == "__main__":
    demonstrate_concurrent_processing()
