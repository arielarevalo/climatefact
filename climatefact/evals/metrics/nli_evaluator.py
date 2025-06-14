"""
NLI evaluation pipeline for assessing natural language inference model performance.
Integrates with existing NLI workflows to measure quality using standard classification metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any

from climatefact.workflows.contradiction_detection.nodes.detect_contradictions import call_nli_model

from .nli_metrics import NLIMetrics

logger = logging.getLogger(__name__)


class NLIEvaluator:
    """
    End-to-end evaluator for NLI systems.
    Handles data loading, NLI model invocation, metric calculation, and report generation.
    """

    def __init__(self, metrics_calculator: NLIMetrics | None = None):
        self.metrics_calculator = metrics_calculator or NLIMetrics()
        self.passages_cache = {}

    def load_passages(self, passages_path: str) -> dict[str, str]:
        """
        Load passages from JSONL file and cache them by ID.

        Args:
            passages_path: Path to passages.jsonl file

        Returns:
            Dictionary mapping passage IDs to passage text
        """
        if not Path(passages_path).exists():
            raise FileNotFoundError(f"Passages file not found: {passages_path}")

        logger.info(f"Loading passages from {passages_path}")
        passages = {}

        try:
            with open(passages_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            passage_data = json.loads(line)
                            passage_id = passage_data.get("id")
                            passage_text = passage_data.get("text")

                            if not passage_id:
                                raise ValueError(f"Passage missing 'id' field at line {line_num}")
                            if not passage_text:
                                raise ValueError(f"Passage missing 'text' field at line {line_num}")

                            passages[passage_id] = passage_text

                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to load passages from {passages_path}: {e}") from e

        if not passages:
            raise ValueError(f"No passages loaded from {passages_path}")

        logger.info(f"Loaded {len(passages)} passages")
        self.passages_cache = passages
        return passages

    def load_gold_set(self, gold_set_path: str) -> list[dict[str, Any]]:
        """
        Load gold standard dataset from JSONL file.

        Args:
            gold_set_path: Path to gold_set.jsonl file

        Returns:
            List of gold standard entries
        """
        if not Path(gold_set_path).exists():
            raise FileNotFoundError(f"Gold set file not found: {gold_set_path}")

        logger.info(f"Loading gold set from {gold_set_path}")
        gold_entries = []

        try:
            with open(gold_set_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            gold_entries.append(entry)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to load gold set from {gold_set_path}: {e}") from e

        if not gold_entries:
            raise ValueError(f"No entries loaded from gold set: {gold_set_path}")

        logger.info(f"Loaded {len(gold_entries)} gold set entries")
        return gold_entries

    def filter_nli_entries(self, gold_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter gold set entries to only include those with non-null entailment values.

        Args:
            gold_entries: List of all gold set entries

        Returns:
            List of entries suitable for NLI evaluation
        """
        nli_entries = []

        for entry in gold_entries:
            entailment = entry.get("entailment")
            claim = entry.get("claim")
            evidence = entry.get("evidence")

            # Only include entries with non-null entailment
            if entailment is not None and claim and evidence:
                nli_entries.append(entry)

        if not nli_entries:
            raise ValueError("No entries with non-null entailment found in gold set")

        logger.info(f"Filtered to {len(nli_entries)} entries with non-null entailment")
        return nli_entries

    def prepare_nli_test_cases(self, nli_entries: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        """
        Prepare test cases for NLI evaluation.

        Args:
            nli_entries: Filtered gold set entries with non-null entailment

        Returns:
            List of (claim, passage_text, expected_label) tuples
        """
        if not self.passages_cache:
            raise RuntimeError("Passages not loaded - call load_passages first")

        test_cases = []

        for entry in nli_entries:
            claim = entry.get("claim", "").strip()
            evidence_id = entry.get("evidence", "")
            expected_label = entry.get("entailment", "").strip()

            if not claim:
                raise ValueError(f"Empty claim in entry: {entry}")
            if not expected_label:
                raise ValueError(f"Empty entailment label in entry: {entry}")
            if not evidence_id:
                raise ValueError(f"Empty evidence ID in entry: {entry}")

            # Create test case for this evidence passage
            if evidence_id not in self.passages_cache:
                raise ValueError(f"Evidence passage '{evidence_id}' not found in passages cache")

            passage_text = self.passages_cache[evidence_id]
            if not passage_text.strip():
                raise ValueError(f"Empty passage text for evidence ID: {evidence_id}")

            test_cases.append((claim, passage_text, expected_label))

        if not test_cases:
            raise ValueError("No valid test cases generated from NLI entries")

        logger.info(f"Generated {len(test_cases)} NLI test cases")
        return test_cases

    def run_nli_evaluation(self, test_cases: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
        """
        Run NLI model on test cases and collect predictions.

        Args:
            test_cases: List of (claim, passage_text, expected_label) tuples

        Returns:
            List of (predicted_label, true_label) tuples
        """
        predictions = []

        for i, (claim, passage_text, expected_label) in enumerate(test_cases):
            logger.info(f"Running NLI model on test case {i + 1}/{len(test_cases)}")

            try:
                # Call the NLI model (passage is premise, claim is hypothesis)
                predicted_label = call_nli_model(premise=passage_text, hypothesis=claim)

                if not predicted_label:
                    raise RuntimeError(f"NLI model returned empty prediction for test case {i + 1}")

                predictions.append((predicted_label, expected_label))

            except Exception as e:
                raise RuntimeError(
                    f"NLI model failed on test case {i + 1} "
                    f"(claim: '{claim[:50]}...', passage: '{passage_text[:50]}...'): {e}"
                ) from e

        if not predictions:
            raise RuntimeError("No predictions generated from NLI evaluation")

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def evaluate_nli_performance(
        self, gold_set_path: str, passages_path: str, max_queries: int | None = None
    ) -> dict[str, Any]:
        """
        Run complete NLI evaluation pipeline.

        Args:
            gold_set_path: Path to gold standard dataset
            passages_path: Path to passages file
            max_queries: Maximum number of queries to evaluate (None for all)

        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting NLI evaluation pipeline")

        # Load data
        self.load_passages(passages_path)
        gold_entries = self.load_gold_set(gold_set_path)

        # Filter for NLI entries
        nli_entries = self.filter_nli_entries(gold_entries)

        # Limit queries if specified
        if max_queries is not None and len(nli_entries) > max_queries:
            nli_entries = nli_entries[:max_queries]
            logger.info(f"Limited evaluation to {len(nli_entries)} entries")

        # Prepare test cases
        test_cases = self.prepare_nli_test_cases(nli_entries)

        # Run NLI evaluation
        predictions = self.run_nli_evaluation(test_cases)

        # Calculate metrics
        results = self.metrics_calculator.evaluate_multiple_predictions(predictions)

        logger.info("NLI evaluation pipeline completed successfully")
        return results

    def generate_detailed_report(self, results: dict[str, Any], output_path: str) -> None:
        """
        Generate detailed evaluation report.

        Args:
            results: Results from evaluate_nli_performance
            output_path: Path to save the report
        """
        if not results:
            raise ValueError("No results provided for report generation")

        try:
            summary = self.metrics_calculator.get_metrics_summary(results)

            # Create full report
            report = f"""
NLI MODEL EVALUATION REPORT
==========================
Generated at: {Path(output_path).name}

{summary}

DETAILED ANALYSIS
================
Label Distribution (Predicted): {results.get("label_distribution", {}).get("predicted", {})}
Label Distribution (True): {results.get("label_distribution", {}).get("true", {})}

This report shows the performance of the NLI model on the climate fact-checking dataset.
The model was evaluated on claims vs. evidence passages to determine entailment relationships.
"""

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"Detailed report saved to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to generate report: {e}") from e
