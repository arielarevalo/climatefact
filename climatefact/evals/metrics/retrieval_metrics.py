"""
Retrieval evaluation metrics for assessing quality of retrieval, embeddings, and reranking.
Implements recall@k, precision@k, MRR@k, and nDCG@k metrics.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Comprehensive retrieval metrics calculator for evaluating retrieval systems.

    Supports evaluation of:
    - Retrieval quality (recall@k, precision@k)
    - Ranking quality (MRR@k, nDCG@k)
    - Embedding effectiveness
    - Reranking performance
    """

    def __init__(self):
        self.metrics_history = []

    def calculate_recall_at_k(self, retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate recall@k: fraction of relevant items retrieved in top-k results.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevant_ids: Set of IDs that are relevant for the query
            k: Number of top results to consider

        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0

        top_k_retrieved = set(retrieved_ids[:k])
        relevant_retrieved = top_k_retrieved.intersection(relevant_ids)

        recall = len(relevant_retrieved) / len(relevant_ids)
        return recall

    def calculate_precision_at_k(self, retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate precision@k: fraction of retrieved items that are relevant in top-k results.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevant_ids: Set of IDs that are relevant for the query
            k: Number of top results to consider

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0

        top_k_retrieved = retrieved_ids[:k]
        if not top_k_retrieved:
            return 0.0

        relevant_retrieved = sum(1 for item_id in top_k_retrieved if item_id in relevant_ids)
        precision = relevant_retrieved / len(top_k_retrieved)

        return precision

    def calculate_f1_at_k(self, retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate F1@k: harmonic mean of precision@k and recall@k.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevant_ids: Set of IDs that are relevant for the query
            k: Number of top results to consider

        Returns:
            F1@k score (0.0 to 1.0)
        """
        if not relevant_ids or k == 0:
            return 0.0

        # Calculate precision@k and recall@k
        precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)

        # F1 is harmonic mean of precision and recall
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def calculate_mrr_at_k(self, retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank@k: reciprocal of rank of first relevant item.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevant_ids: Set of IDs that are relevant for the query
            k: Number of top results to consider

        Returns:
            MRR@k score (0.0 to 1.0)
        """
        top_k_retrieved = retrieved_ids[:k]

        for rank, item_id in enumerate(top_k_retrieved, 1):
            if item_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def calculate_dcg_at_k(self, retrieved_ids: list[str], relevance_scores: dict[str, float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@k.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevance_scores: Dict mapping passage IDs to relevance scores
            k: Number of top results to consider

        Returns:
            DCG@k score
        """
        dcg = 0.0
        top_k_retrieved = retrieved_ids[:k]

        for rank, item_id in enumerate(top_k_retrieved, 1):
            relevance = relevance_scores.get(item_id, 0.0)
            if rank == 1:
                dcg += relevance
            else:
                dcg += relevance / math.log2(rank)

        return dcg

    def calculate_ndcg_at_k(self, retrieved_ids: list[str], relevance_scores: dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.

        Args:
            retrieved_ids: List of retrieved passage IDs (ordered by relevance)
            relevance_scores: Dict mapping passage IDs to relevance scores
            k: Number of top results to consider

        Returns:
            nDCG@k score (0.0 to 1.0)
        """
        # Calculate DCG@k for retrieved items
        dcg_k = self.calculate_dcg_at_k(retrieved_ids, relevance_scores, k)

        # Calculate ideal DCG@k (IDCG@k)
        # Sort all items by relevance score in descending order
        sorted_items = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        ideal_retrieved_ids = [item_id for item_id, _ in sorted_items]
        idcg_k = self.calculate_dcg_at_k(ideal_retrieved_ids, relevance_scores, k)

        # Avoid division by zero
        if idcg_k == 0.0:
            return 0.0

        ndcg_k = dcg_k / idcg_k
        return ndcg_k

    def evaluate_single_query(
        self, retrieved_passages: list[dict[str, Any]], relevant_passage_id: str, k_values: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate retrieval performance for a single query.

        Args:
            retrieved_passages: List of retrieved passage dictionaries
            relevant_passage_id: ID of the relevant passage for this query, or "__CONTROL_CASE__" for control cases
            k_values: List of k values to evaluate

        Returns:
            Dictionary of metrics for each k value
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        # Handle control cases
        if relevant_passage_id == "__CONTROL_CASE__":
            return self._evaluate_control_case(retrieved_passages, k_values)

        retrieved_ids = [passage["id"] for passage in retrieved_passages]
        relevant_ids = {relevant_passage_id}

        # Create relevance scores (binary: 1 for relevant, 0 for irrelevant)
        relevance_scores = {}
        for passage in retrieved_passages:
            relevance_scores[passage["id"]] = 1.0 if passage["id"] == relevant_passage_id else 0.0

        results = {}

        for k in k_values:
            results[f"k_{k}"] = {
                "recall": self.calculate_recall_at_k(retrieved_ids, relevant_ids, k),
                "precision": self.calculate_precision_at_k(retrieved_ids, relevant_ids, k),
                "f1": self.calculate_f1_at_k(retrieved_ids, relevant_ids, k),
                "mrr": self.calculate_mrr_at_k(retrieved_ids, relevant_ids, k),
                "ndcg": self.calculate_ndcg_at_k(retrieved_ids, relevance_scores, k),
            }

        return results

    def evaluate_multiple_queries(
        self, queries_results: list[tuple[list[dict[str, Any]], str]], k_values: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate retrieval performance across multiple queries and calculate averages.

        Args:
            queries_results: List of tuples (retrieved_passages, relevant_passage_id)
            k_values: List of k values to evaluate

        Returns:
            Dictionary of averaged metrics for each k value
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        if not queries_results:
            return {}

        # Collect all individual results
        all_results = []
        for retrieved_passages, relevant_passage_id in queries_results:
            single_result = self.evaluate_single_query(retrieved_passages, relevant_passage_id, k_values)
            all_results.append(single_result)

        # Calculate averages
        averaged_results = {}
        for k in k_values:
            k_key = f"k_{k}"

            # Initialize sums
            sums = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "mrr": 0.0, "ndcg": 0.0}

            # Sum up all values
            for result in all_results:
                if k_key in result:
                    for metric in sums:
                        sums[metric] += result[k_key][metric]

            # Calculate averages
            num_queries = len(all_results)
            averaged_results[k_key] = {metric: sums[metric] / num_queries for metric in sums}

        # Store in history
        self.metrics_history.append(averaged_results)

        return averaged_results

    def compare_retrieval_methods(self, method_results: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
        """
        Compare multiple retrieval methods across metrics.

        Args:
            method_results: Dict mapping method names to their evaluation results

        Returns:
            Comparison analysis with best performing methods per metric
        """
        comparison = {"method_comparison": {}, "best_methods": {}, "performance_summary": {}}

        if not method_results:
            return comparison

        # Get all k values and metrics
        first_method = next(iter(method_results.values()))
        k_values = list(first_method.keys())
        metrics = list(first_method[k_values[0]].keys()) if k_values else []

        # Compare each metric at each k
        for k in k_values:
            comparison["method_comparison"][k] = {}
            comparison["best_methods"][k] = {}

            for metric in metrics:
                method_scores = {}
                for method_name, results in method_results.items():
                    if k in results and metric in results[k]:
                        method_scores[method_name] = results[k][metric]

                comparison["method_comparison"][k][metric] = method_scores

                if method_scores:
                    best_method = max(method_scores.items(), key=lambda x: x[1])
                    comparison["best_methods"][k][metric] = {"method": best_method[0], "score": best_method[1]}

        # Overall performance summary
        method_wins = {method: 0 for method in method_results}
        total_comparisons = 0

        for k in k_values:
            for metric in metrics:
                if k in comparison["best_methods"] and metric in comparison["best_methods"][k]:
                    best_method = comparison["best_methods"][k][metric]["method"]
                    method_wins[best_method] += 1
                    total_comparisons += 1

        comparison["performance_summary"] = {
            "method_wins": method_wins,
            "total_comparisons": total_comparisons,
            "win_percentages": {
                method: (wins / total_comparisons * 100) if total_comparisons > 0 else 0
                for method, wins in method_wins.items()
            },
        }

        return comparison

    def get_metrics_summary(self, results: dict[str, dict[str, float]]) -> str:
        """
        Generate a human-readable summary of metrics results.

        Args:
            results: Results dictionary from evaluation

        Returns:
            Formatted string summary
        """
        if not results:
            return "No results to summarize."

        summary_lines = ["Retrieval Metrics Summary", "=" * 30]

        for k_key, metrics in results.items():
            k_value = k_key.replace("k_", "")
            summary_lines.append(f"\nTop-{k_value} Results:")
            summary_lines.append(f"  Recall@{k_value}:    {metrics['recall']:.4f}")
            summary_lines.append(f"  Precision@{k_value}: {metrics['precision']:.4f}")
            summary_lines.append(f"  F1@{k_value}:        {metrics['f1']:.4f}")
            summary_lines.append(f"  MRR@{k_value}:       {metrics['mrr']:.4f}")
            summary_lines.append(f"  nDCG@{k_value}:      {metrics['ndcg']:.4f}")

        return "\n".join(summary_lines)

    def _evaluate_control_case(
        self, retrieved_passages: list[dict[str, Any]], k_values: list[int] | None = None
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate a control case query where no results should be returned.

        Args:
            retrieved_passages: List of retrieved passage dictionaries
            k_values: List of k values to evaluate

        Returns:
            Dictionary of metrics for each k value
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        num_retrieved = len(retrieved_passages)
        results = {}

        for k in k_values:
            # For control cases, perfect scores are:
            # - recall: 1.0 if no results retrieved (as expected), 0.0 if any results
            # - precision: 1.0 if no results retrieved, 0.0 if any results
            # - f1: 1.0 if no results retrieved, 0.0 if any results
            # - mrr: 1.0 if no results retrieved, 0.0 if any results
            # - ndcg: 1.0 if no results retrieved, 0.0 if any results

            perfect_score = 1.0 if num_retrieved == 0 else 0.0

            results[f"k_{k}"] = {
                "recall": perfect_score,
                "precision": perfect_score,
                "f1": perfect_score,
                "mrr": perfect_score,
                "ndcg": perfect_score,
            }

        return results
