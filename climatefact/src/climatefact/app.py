import streamlit as st
from langchain_core.runnables import RunnableConfig

from climatefact.workflows.contradiction_detection import graph

# Constant path for passages JSONL file
PASSAGES_JSONL_PATH = "../data/passages.jsonl"
CONCEPT_INDEX_PATH = "../data/concept_index.json"


def main():
    st.set_page_config(
        page_title="ClimateFact - Contradiction Detection",
        page_icon="🌍",
        layout="wide",
    )

    st.title("🌍 ClimateFact - Climate Statement Contradiction Detection")
    st.markdown("Enter a climate-related statement to check for contradictions against scientific evidence.")

    # Text input
    input_text = st.text_area(
        "Climate Statement",
        placeholder="Enter your climate-related statement here...",
        height=150,
    )

    # Submit button
    if st.button("🔍 Check for Contradictions", type="primary"):
        if not input_text.strip():
            st.error("Please enter a statement to analyze.")
            return

        # Configuration for the graph
        config = RunnableConfig(
            configurable={
                "passages_jsonl_path": PASSAGES_JSONL_PATH,
                "concept_index_path": CONCEPT_INDEX_PATH,
            }
        )

        # Initial state
        initial_state = {
            "input_text": input_text.strip(),
            "queries": None,
            "retrieved_data_for_queries": None,
            "contradiction_results": None,
            "report": None,
        }

        # Run the graph with spinner
        with st.spinner("🔄 Analyzing statement for contradictions..."):
            try:
                result = graph.invoke(initial_state, config=config)

                # Display results
                st.success("✅ Analysis complete!")

                if result.get("report"):
                    st.markdown(result["report"])
                else:
                    st.warning("No report generated. Please check your configuration and try again.")

                # Show raw results in expander for debugging
                with st.expander("🔍 View Raw Results"):
                    st.json(result)

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {e!s}")
                st.error("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
