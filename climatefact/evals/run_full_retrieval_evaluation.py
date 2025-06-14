"""
Evaluation runner for the complete retrieval subgraph.
Tests the full retrieval pipeline that populates retrieved_data_for_queries.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from climatefact.evals.metrics.retrieval_evaluator import RetrievalEvaluator

from climatefact.workflows.contradiction_detection.subgraphs.retrieval import graph as retrieval_graph
from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("full_pipeline_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class FullPipelineEvaluationRunner:
    """
    Runner for evaluating the complete retrieval subgraph pipeline.
    Tests what actually ends up in retrieved_data_for_queries.
    """

    def __init__(self, base_data_path: str | None = None):
        # Use relative paths from the evals directory
        self.base_data_path = Path(base_data_path) if base_data_path else Path(__file__).parent.parent.parent / "data"
        self.evaluator = RetrievalEvaluator()

        # Standard file paths
        self.gold_set_path = self.base_data_path / "evaluation" / "gold_set.jsonl"
        self.passages_path = self.base_data_path / "passages.jsonl"
        self.concept_index_path = self.base_data_path / "concept_index.json"

        # Results storage
        self.results = {}

    def verify_data_files(self) -> bool:
        """
        Verify that all required data files exist.

        Returns:
            True if all files exist, False otherwise
        """
        required_files = [
            self.gold_set_path,
            self.passages_path,
            self.concept_index_path,
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False

        logger.info("All required data files found")
        return True

    def load_gold_standard_data(self) -> list[dict[str, Any]]:
        """
        Load and validate gold standard data from JSONL format.

        Returns:
            List of gold standard entries
        """
        if not self.gold_set_path.exists():
            logger.error(f"Gold set file not found: {self.gold_set_path}")
            return []

        gold_entries = []

        try:
            with open(self.gold_set_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            entry = json.loads(line)
                            gold_entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

        except Exception as e:
            logger.error(f"Error loading gold set: {e}")
            return []

        logger.info(f"Loaded {len(gold_entries)} gold standard entries from JSONL")
        return gold_entries

    def run_full_retrieval_pipeline(self, queries: list[str], k_value: int = 10) -> list[list[dict[str, Any]]]:
        """
        Run the complete retrieval subgraph pipeline with specified k value.

        Args:
            queries: List of query strings
            k_value: Number of results to retrieve (used for both hybrid and semantic search)

        Returns:
            Retrieved data for queries (output of full pipeline)
        """
        logger.info(f"Running full retrieval pipeline for {len(queries)} queries with k={k_value}")

        # Prepare initial state
        initial_state: RetrievalState = {
            "queries": queries,
            "regex_retrieved_data": None,
            "ner_retrieved_data": None,
            "semantic_retrieved_data": None,
            "hybrid_retrieved_data": None,
            "retrieved_data_for_queries": None,
        }

        # Prepare configuration with the specified k value
        from langchain_core.runnables import RunnableConfig

        config = RunnableConfig(
            configurable={
                "passages_jsonl_path": str(self.passages_path),
                "concept_index_path": str(self.concept_index_path),
                "hybrid_retrieval_top_k": k_value,
                "semantic_search_top_k": k_value,
            }
        )

        try:
            # Run the complete retrieval subgraph
            logger.info(
                f"Executing retrieval subgraph with hybrid_retrieval_top_k={k_value}, semantic_search_top_k={k_value}"
            )
            final_state = retrieval_graph.invoke(initial_state, config)

            # Extract the final results
            retrieved_data = final_state.get("retrieved_data_for_queries", [])
            logger.info(f"Pipeline completed. Got results for {len(retrieved_data)} queries")

            # Log summary of results
            for i, query_results in enumerate(retrieved_data):
                logger.debug(f"Query {i}: {len(query_results)} passages retrieved")
                if query_results and i < 3:  # Only log first few for brevity
                    first_passage = query_results[0]
                    passage_id = first_passage.get("id", "NO_ID")
                    logger.debug(f"  First passage ID: {passage_id}")

            return retrieved_data

        except Exception as e:
            logger.error(f"Error running retrieval pipeline: {e}")
            return []

    def evaluate_full_pipeline(
        self, gold_entries: list[dict[str, Any]], k_values: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the complete retrieval pipeline for each k value.

        Args:
            gold_entries: Gold standard entries
            k_values: List of k values to evaluate

        Returns:
            Evaluation results for each k value
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        logger.info("Starting full pipeline evaluation")

        # Extract queries from gold standard
        queries = [entry.get("claim", "") for entry in gold_entries if entry.get("claim")]

        if not queries:
            logger.error("No valid queries found in gold standard data")
            return {}

        logger.info(f"Evaluating {len(queries)} queries for k values: {k_values}")

        # Store results for all k values
        all_k_results = {}

        # Evaluate each k value separately
        for k in k_values:
            logger.info(f"=== EVALUATING K={k} ===")

            # Run the full retrieval pipeline with this k value, passing gold entries for control case detection
            all_retrieved_passages = self.run_full_retrieval_pipeline_with_control_cases(gold_entries, k_value=k)

            if not all_retrieved_passages:
                logger.error(f"No results from retrieval pipeline for k={k}")
                continue

            if len(all_retrieved_passages) != len(queries):
                logger.warning(
                    f"Mismatch for k={k}: {len(queries)} queries but {len(all_retrieved_passages)} result sets"
                )

            # Evaluate using the evaluator (but only for this specific k value)
            k_results = self.evaluator.evaluate_retrieval_method(
                method_name=f"full_pipeline_k{k}",
                gold_set_path=str(self.gold_set_path),
                retrieved_results=all_retrieved_passages,
                k_values=[k],  # Only evaluate this specific k
            )

            # Extract the results for this k and store them
            if k_results and f"k_{k}" in k_results:
                all_k_results[f"k_{k}"] = k_results[f"k_{k}"]
                logger.info(f"K={k} results: {k_results[f'k_{k}']}")

        logger.info("Completed evaluation for all k values")
        return all_k_results

    def debug_pipeline_results(self, gold_entries: list[dict[str, Any]], max_queries: int = 5) -> None:
        """
        Debug the pipeline results to understand what's being retrieved.

        Args:
            gold_entries: Gold standard entries
            max_queries: Maximum number of queries to debug (for performance)
        """
        logger.info("=== DEBUGGING FULL PIPELINE RESULTS ===")

        # Limit queries for debugging
        debug_entries = gold_entries[:max_queries]
        queries = [entry.get("claim", "") for entry in debug_entries if entry.get("claim")]

        # Use k=10 for debugging to see more results
        retrieved_data = self.run_full_retrieval_pipeline(queries, k_value=10)

        for i, (entry, query_results) in enumerate(zip(debug_entries, retrieved_data, strict=False)):
            expected_evidence_id = entry.get("evidence", "")
            logger.info(f"\n--- Query {i + 1} ---")
            logger.info(f"Claim: {entry.get('claim', '')[:100]}...")
            logger.info(f"Expected evidence ID: {expected_evidence_id}")
            logger.info(f"Retrieved {len(query_results)} passages")

            # Check if expected evidence is in results
            found_evidence = False
            for j, passage in enumerate(query_results):
                passage_id = passage.get("id", "NO_ID")
                logger.info(f"  {j + 1}. Passage ID: {passage_id}")
                if passage_id == expected_evidence_id:
                    found_evidence = True
                    logger.info(f"     ✓ FOUND EXPECTED EVIDENCE at position {j + 1}")

            if not found_evidence and expected_evidence_id:
                logger.info(f"     ✗ Expected evidence {expected_evidence_id} not found")

        logger.info("=== DEBUG COMPLETE ===")

    def run_comprehensive_evaluation(
        self, k_values: list[int] | None = None, max_queries: int | None = None
    ) -> dict[str, Any]:
        """
        Run comprehensive evaluation of the full retrieval pipeline.

        Args:
            k_values: List of k values to evaluate
            max_queries: Maximum number of queries to evaluate (None for all)

        Returns:
            Complete evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        logger.info("Starting comprehensive evaluation of full retrieval pipeline")

        # Verify data files exist
        if not self.verify_data_files():
            logger.error("Data file verification failed")
            return {"error": "Missing required data files"}

        # Load gold standard data
        gold_entries = self.load_gold_standard_data()
        if not gold_entries:
            logger.error("No gold standard data loaded")
            return {"error": "No gold standard data"}

        # Limit queries if specified
        if max_queries is not None:
            gold_entries = gold_entries[:max_queries]
            logger.info(f"Limited evaluation to {len(gold_entries)} queries")

        # Debug pipeline results first (only first 5 for performance)
        self.debug_pipeline_results(gold_entries, max_queries=5)

        # Run evaluation
        pipeline_results = self.evaluate_full_pipeline(gold_entries, k_values)

        if not pipeline_results:
            logger.error("No evaluation results generated")
            return {"error": "Evaluation failed"}

        # Compile comprehensive results
        comprehensive_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "gold_set_size": len(gold_entries),
                "methods_evaluated": ["full_pipeline"],
                "k_values": k_values,
                "pipeline_components": [
                    "retrieve_by_regex",
                    "retrieve_by_ner",
                    "retrieve_by_semantic_search",
                    "combine_and_semantic_search",
                    "deduplicate_results",
                ],
            },
            "method_results": {"full_pipeline": pipeline_results},
        }

        return comprehensive_results

    def save_results(self, results: dict[str, Any], output_dir: str = "evals/reports"):
        """
        Save evaluation results to files.

        Args:
            results: Evaluation results
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        main_file = output_path / f"full_pipeline_evaluation_{timestamp}.json"
        try:
            with open(main_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {main_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def run_full_retrieval_pipeline_with_control_cases(
        self, gold_entries: list[dict[str, Any]], k_value: int = 10
    ) -> list[list[dict[str, Any]]]:
        """
        Run the complete retrieval pipeline, handling control cases by returning empty results.

        Args:
            gold_entries: List of gold standard entries (containing queries and control case info)
            k_value: Number of results to retrieve for non-control cases

        Returns:
            Retrieved data for queries, with empty lists for control cases
        """
        logger.info(f"Running full retrieval pipeline for {len(gold_entries)} entries with k={k_value}")

        results = []
        regular_queries = []
        regular_indices = []

        # First pass: identify control cases and regular queries
        for i, gold_entry in enumerate(gold_entries):
            evidence = gold_entry.get("evidence")
            entailment = gold_entry.get("entailment")

            if evidence is None and entailment is None:
                # Control case - should return empty results
                logger.info(f"Control case detected for entry {gold_entry.get('id', 'unknown')}")
                results.append([])  # Empty results for control case
            else:
                # Regular query
                query = gold_entry.get("claim", "")
                if query:  # Only add non-empty queries
                    regular_queries.append(query)
                    regular_indices.append(i)
                results.append(None)  # Placeholder

        # Second pass: run pipeline for regular queries if any
        if regular_queries:
            logger.info(f"Running pipeline for {len(regular_queries)} regular queries")
            regular_results = self.run_full_retrieval_pipeline(regular_queries, k_value)

            # Fill in results for regular queries
            for idx, result in zip(regular_indices, regular_results, strict=False):
                results[idx] = result

        logger.info(
            f"Pipeline completed. Control cases: {len(gold_entries) - len(regular_queries)}, "
            f"Regular queries: {len(regular_queries)}"
        )
        return results


def main():
    """Run the full pipeline evaluation."""
    runner = FullPipelineEvaluationRunner()

    # Start with a smaller subset for testing (20 queries)
    # Change max_queries=None to evaluate all 847 queries
    results = runner.run_comprehensive_evaluation(max_queries=400)

    if "error" not in results:
        # Save results
        runner.save_results(results)

        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        method_results = results.get("method_results", {}).get("full_pipeline", {})

        for k_key, metrics in method_results.items():
            logger.info(f"{k_key.upper()}:")
            for metric_name, score in metrics.items():
                logger.info(f"  {metric_name}: {score:.4f}")
    else:
        logger.error(f"Evaluation failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
