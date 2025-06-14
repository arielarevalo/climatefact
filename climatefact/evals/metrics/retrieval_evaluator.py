"""
Evaluation pipeline for assessing retrieval system performance.
Integrates with existing retrieval workflows to measure quality using standard metrics.
"""

import json
import logging
import os
from typing import Any

from .retrieval_metrics import RetrievalMetrics

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    End-to-end evaluator for retrieval systems.
    Handles data loading, metric calculation, and report generation.
    """

    def __init__(self, metrics_calculator: RetrievalMetrics | None = None):
        self.metrics = metrics_calculator or RetrievalMetrics()
        self.evaluation_results = {}

    def load_gold_set(self, gold_set_path: str) -> list[dict[str, Any]]:
        """
        Load the gold standard dataset.

        Args:
            gold_set_path: Path to the gold set JSON/JSONL file

        Returns:
            List of gold standard entries
        """
        gold_entries = []

        if not os.path.exists(gold_set_path):
            logger.error(f"Gold set file not found: {gold_set_path}")
            return gold_entries

        try:
            with open(gold_set_path, encoding="utf-8") as f:
                # Check if it's a JSON array or JSONL format
                if gold_set_path.endswith(".json"):
                    # JSON array format
                    data = json.load(f)
                    if isinstance(data, list):
                        gold_entries = data
                    else:
                        logger.error("Expected JSON array format for gold set")
                        return []
                else:
                    # JSONL format
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            gold_entries.append(entry)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Error reading gold set file: {e}")

        logger.info(f"Loaded {len(gold_entries)} gold standard entries")
        return gold_entries

    def prepare_query_data(self, gold_entries: list[dict[str, Any]]) -> list[str]:
        """
        Extract queries from gold standard entries.

        Args:
            gold_entries: List of gold standard entries

        Returns:
            List of query strings
        """
        queries = []
        for entry in gold_entries:
            claim = entry.get("claim", "")
            if claim:
                queries.append(claim)
            else:
                logger.warning(f"Empty claim found in entry {entry.get('id', 'unknown')}")

        return queries

    def create_evaluation_pairs(
        self, gold_entries: list[dict[str, Any]], retrieved_results: list[list[dict[str, Any]]]
    ) -> list[tuple[list[dict[str, Any]], str]]:
        """
        Create pairs of retrieved results and relevant passage IDs for evaluation.

        Args:
            gold_entries: List of gold standard entries
            retrieved_results: List of retrieved passages for each query

        Returns:
            List of tuples (retrieved_passages, relevant_passage_id)
        """
        evaluation_pairs = []

        if len(gold_entries) != len(retrieved_results):
            logger.warning(f"Mismatch: {len(gold_entries)} gold entries vs {len(retrieved_results)} retrieved results")
            min_length = min(len(gold_entries), len(retrieved_results))
            gold_entries = gold_entries[:min_length]
            retrieved_results = retrieved_results[:min_length]

        for gold_entry, retrieved_passages in zip(gold_entries, retrieved_results, strict=False):
            relevant_id = gold_entry.get("evidence")  # Don't provide default, let it be None if missing
            entailment = gold_entry.get("entailment")  # Don't provide default, let it be None if missing

            if relevant_id:
                # Normal case with evidence
                evaluation_pairs.append((retrieved_passages, relevant_id))
            elif relevant_id is None and entailment is None:
                # Control case - should return empty results
                evaluation_pairs.append((retrieved_passages, "__CONTROL_CASE__"))
                logger.info(f"Including control case {gold_entry.get('id', 'unknown')} in evaluation")
            else:
                logger.warning(f"No evidence ID found for entry {gold_entry.get('id', 'unknown')}")

        return evaluation_pairs

    def evaluate_retrieval_method(
        self,
        method_name: str,
        gold_set_path: str,
        retrieved_results: list[list[dict[str, Any]]],
        k_values: list[int] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate a single retrieval method.

        Args:
            method_name: Name of the retrieval method
            gold_set_path: Path to gold standard dataset
            retrieved_results: Retrieved passages for each query
            k_values: List of k values to evaluate

        Returns:
            Evaluation results dictionary
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        logger.info(f"Evaluating retrieval method: {method_name}")

        # Load gold standard data
        gold_entries = self.load_gold_set(gold_set_path)
        if not gold_entries:
            logger.error("No gold standard data loaded")
            return {}

        # Create evaluation pairs
        evaluation_pairs = self.create_evaluation_pairs(gold_entries, retrieved_results)
        if not evaluation_pairs:
            logger.error("No valid evaluation pairs created")
            return {}

        # Calculate metrics
        results = self.metrics.evaluate_multiple_queries(evaluation_pairs, k_values)

        # Store results
        self.evaluation_results[method_name] = results

        logger.info(f"Completed evaluation for {method_name}")
        return results

    def compare_methods(self, method_results: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
        """
        Compare multiple retrieval methods.

        Args:
            method_results: Dictionary mapping method names to their results

        Returns:
            Comparison analysis
        """
        return self.metrics.compare_retrieval_methods(method_results)

    def generate_detailed_report(
        self, method_results: dict[str, dict[str, dict[str, float]]], output_path: str
    ) -> None:
        """
        Generate a detailed evaluation report.

        Args:
            method_results: Dictionary mapping method names to their results
            output_path: Path to save the report
        """
        report = {
            "evaluation_summary": {
                "methods_evaluated": list(method_results.keys()),
                "metrics_calculated": ["recall", "precision", "mrr", "ndcg"],
                "k_values": [],
            },
            "individual_results": method_results,
            "comparison_analysis": self.compare_methods(method_results),
            "recommendations": self._generate_recommendations(method_results),
        }

        # Extract k values from first method
        if method_results:
            first_method = next(iter(method_results.values()))
            report["evaluation_summary"]["k_values"] = [int(k.replace("k_", "")) for k in first_method]

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed report saved to: {output_path}")

    def _generate_recommendations(self, method_results: dict[str, dict[str, dict[str, float]]]) -> list[str]:
        """
        Generate recommendations based on evaluation results.

        Args:
            method_results: Dictionary mapping method names to their results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not method_results:
            return ["No results available for generating recommendations."]

        comparison = self.compare_methods(method_results)
        win_percentages = comparison.get("performance_summary", {}).get("win_percentages", {})

        if win_percentages:
            best_overall = max(win_percentages.items(), key=lambda x: x[1])
            recommendations.append(
                f"Overall best performing method: {best_overall[0]} (wins {best_overall[1]:.1f}% of comparisons)"
            )

        # Analyze specific metrics
        best_methods = comparison.get("best_methods", {})
        if "k_5" in best_methods:  # Focus on k=5 as a good middle ground
            k5_results = best_methods["k_5"]

            if "recall" in k5_results:
                recommendations.append(
                    f"Best recall@5: {k5_results['recall']['method']} ({k5_results['recall']['score']:.4f})"
                )

            if "precision" in k5_results:
                recommendations.append(
                    f"Best precision@5: {k5_results['precision']['method']} ({k5_results['precision']['score']:.4f})"
                )

            if "ndcg" in k5_results:
                recommendations.append(
                    f"Best ranking quality (nDCG@5): {k5_results['ndcg']['method']} ({k5_results['ndcg']['score']:.4f})"
                )

        # Performance thresholds
        for method_name, results in method_results.items():
            if "k_5" in results:
                recall_5 = results["k_5"].get("recall", 0)
                precision_5 = results["k_5"].get("precision", 0)

                if recall_5 < 0.3:
                    recommendations.append(
                        f"Warning: {method_name} has low recall@5 ({recall_5:.3f}). Consider improving coverage."
                    )

                if precision_5 < 0.2:
                    recommendations.append(
                        f"Warning: {method_name} has low precision@5 ({precision_5:.3f}). "
                        "Consider improving relevance filtering."
                    )

        return recommendations

    def print_results_summary(self, method_results: dict[str, dict[str, dict[str, float]]]) -> None:
        """
        Print a formatted summary of results to console.

        Args:
            method_results: Dictionary mapping method names to their results
        """
        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION RESULTS SUMMARY")
        print("=" * 60)

        for method_name, results in method_results.items():
            print(f"\n{method_name.upper()}:")
            print("-" * len(method_name))
            summary = self.metrics.get_metrics_summary(results)
            print(summary)

        # Print comparison if multiple methods
        if len(method_results) > 1:
            print("\n" + "=" * 60)
            print("METHOD COMPARISON")
            print("=" * 60)

            comparison = self.compare_methods(method_results)
            win_percentages = comparison.get("performance_summary", {}).get("win_percentages", {})

            print("\nOverall Performance (% of metrics where method performed best):")
            for method, percentage in sorted(win_percentages.items(), key=lambda x: x[1], reverse=True):
                print(f"  {method}: {percentage:.1f}%")

        print("\n" + "=" * 60)
