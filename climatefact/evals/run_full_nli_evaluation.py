#!/usr/bin/env python3
"""
Full NLI evaluation runner for the climate fact-checking system.
Tests the NLI model performance on the gold standard dataset.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics.nli_evaluator import NLIEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/nli_evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class FullNLIEvaluationRunner:
    """
    Runner for evaluating the complete NLI model performance.
    Tests how well the model classifies entailment relationships.
    """

    def __init__(self, base_data_path: str | None = None):
        # Use relative paths from the evals directory
        self.base_data_path = base_data_path or str(Path(__file__).parent.parent.parent / "data")
        self.gold_set_path = f"{self.base_data_path}/evaluation/gold_set.jsonl"
        self.passages_path = f"{self.base_data_path}/passages.jsonl"

        # Create output directories relative to this file
        self.reports_dir = Path(__file__).parent / "reports"
        self.logs_dir = Path(__file__).parent / "logs"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluator
        self.evaluator = NLIEvaluator()

    def verify_data_files(self) -> bool:
        """
        Verify that all required data files exist.

        Returns:
            True if all files exist, raises FileNotFoundError otherwise
        """
        required_files = [(self.gold_set_path, "Gold set file"), (self.passages_path, "Passages file")]

        for file_path, description in required_files:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"{description} not found at: {file_path}")

        logger.info("All required data files verified")
        return True

    def analyze_gold_set_entailment_distribution(self) -> dict[str, Any]:
        """
        Analyze the distribution of entailment labels in the gold set.

        Returns:
            Analysis of entailment label distribution
        """
        logger.info("Analyzing gold set entailment distribution...")

        with open(self.gold_set_path, encoding="utf-8") as f:
            total_entries = 0
            entries_with_entailment = 0
            entailment_labels = {}

            for line in f:
                if line.strip():
                    total_entries += 1
                    try:
                        entry = json.loads(line)
                        entailment = entry.get("entailment")

                        if entailment is not None:
                            entries_with_entailment += 1
                            entailment_labels[entailment] = entailment_labels.get(entailment, 0) + 1

                    except json.JSONDecodeError:
                        continue

        analysis = {
            "total_entries": total_entries,
            "entries_with_entailment": entries_with_entailment,
            "entries_without_entailment": total_entries - entries_with_entailment,
            "entailment_distribution": entailment_labels,
            "percentage_with_entailment": ((entries_with_entailment / total_entries * 100) if total_entries > 0 else 0),
        }

        logger.info(f"Gold set analysis: {analysis}")
        return analysis

    def run_full_evaluation(self, max_queries: int | None = None) -> dict[str, Any]:
        """
        Run the complete NLI evaluation pipeline.

        Args:
            max_queries: Maximum number of queries to evaluate (None for all)

        Returns:
            Comprehensive evaluation results
        """
        logger.info("=== STARTING FULL NLI EVALUATION ===")

        # Verify data files
        self.verify_data_files()

        # Analyze gold set
        gold_set_analysis = self.analyze_gold_set_entailment_distribution()

        if gold_set_analysis["entries_with_entailment"] == 0:
            raise ValueError("No entries with entailment labels found in gold set")

        logger.info(f"Found {gold_set_analysis['entries_with_entailment']} entries with entailment labels")

        if max_queries is not None:
            logger.info(f"Limiting evaluation to {max_queries} queries")

        # Run NLI evaluation
        logger.info("Running NLI model evaluation...")
        results = self.evaluator.evaluate_nli_performance(
            gold_set_path=self.gold_set_path, passages_path=self.passages_path, max_queries=max_queries
        )

        # Add gold set analysis to results
        results["gold_set_analysis"] = gold_set_analysis
        results["evaluation_timestamp"] = datetime.now().isoformat()

        return results

    def generate_comprehensive_report(self, results: dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            results: Evaluation results

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"nli_evaluation_comprehensive_{timestamp}.txt"

        try:
            # Generate detailed metrics summary
            metrics_summary = self.evaluator.metrics_calculator.get_metrics_summary(results)

            # Create comprehensive report
            gold_analysis = results.get("gold_set_analysis", {})

            report_content = f"""
COMPREHENSIVE NLI MODEL EVALUATION REPORT
=========================================
Generated: {results.get("evaluation_timestamp", "Unknown")}
Evaluator: Full NLI Evaluation Runner

DATASET ANALYSIS
===============
Total entries in gold set: {gold_analysis.get("total_entries", "N/A")}
Entries with entailment labels: {gold_analysis.get("entries_with_entailment", "N/A")}
Entries without entailment labels: {gold_analysis.get("entries_without_entailment", "N/A")}
Coverage: {gold_analysis.get("percentage_with_entailment", 0):.1f}% of entries have entailment labels

Entailment Label Distribution in Gold Set:
{json.dumps(gold_analysis.get("entailment_distribution", {}), indent=2)}

{metrics_summary}

EVALUATION METHODOLOGY
=====================
1. Loaded passages from JSONL file
2. Filtered gold set entries to only include those with non-null entailment labels
3. For each claim-evidence pair, called the NLI model with:
   - Premise: Evidence passage text
   - Hypothesis: Claim text
4. Compared predicted labels with gold standard labels
5. Calculated standard classification metrics

RECOMMENDATIONS
==============
"""

            # Add recommendations based on results
            accuracy = results.get("accuracy", 0.0)
            macro_f1 = results.get("macro_averages", {}).get("macro_f1", 0.0)

            if accuracy < 0.6:
                report_content += (
                    "\n- Model accuracy is low (<60%). Consider model fine-tuning or different architecture."
                )
            elif accuracy < 0.8:
                report_content += "\n- Model accuracy is moderate (60-80%). There's room for improvement."
            else:
                report_content += "\n- Model accuracy is good (>80%). Performance is acceptable."

            if macro_f1 < 0.5:
                report_content += "\n- Macro F1-score is low. Model struggles with balanced performance across classes."

            # Check for class imbalance issues
            per_class = results.get("per_class_metrics", {})
            worst_class = None
            worst_f1 = 1.0

            for label, metrics in per_class.items():
                f1 = metrics.get("f1_score", 0.0)
                if f1 < worst_f1:
                    worst_f1 = f1
                    worst_class = label

            if worst_class and worst_f1 < 0.3:
                report_content += (
                    f"\n- {worst_class} class performs poorly (F1={worst_f1:.3f}). "
                    "Consider class-specific improvements."
                )

            report_content += f"""

DATA FILES USED
==============
Gold Set: {self.gold_set_path}
Passages: {self.passages_path}

This evaluation tests the NLI model's ability to classify entailment relationships
between climate-related claims and evidence passages.
"""

            # Write report
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Comprehensive report saved to: {report_path}")
            return str(report_path)

        except Exception as e:
            raise RuntimeError(f"Failed to generate comprehensive report: {e}") from e

    def save_results_json(self, results: dict[str, Any]) -> str:
        """
        Save evaluation results as JSON for further analysis.

        Args:
            results: Evaluation results

        Returns:
            Path to saved JSON file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"nli_evaluation_results_{timestamp}.json"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results JSON saved to: {json_path}")
            return str(json_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save results JSON: {e}") from e


def main():
    """
    Main function to run the full NLI evaluation.
    """
    try:
        logger.info("Starting full NLI model evaluation...")

        # Initialize runner
        runner = FullNLIEvaluationRunner()

        # Run evaluation with default limit of 20 queries for testing
        # Change max_queries=None to evaluate all entries with entailment labels
        results = runner.run_full_evaluation(max_queries=400)

        # Generate reports
        report_path = runner.generate_comprehensive_report(results)
        json_path = runner.save_results_json(results)

        # Log summary
        logger.info("=== EVALUATION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Total test cases: {results.get('total_predictions', 0)}")
        logger.info(f"Overall accuracy: {results.get('accuracy', 0.0):.4f}")
        logger.info(f"Macro F1-score: {results.get('macro_averages', {}).get('macro_f1', 0.0):.4f}")
        logger.info(f"Report saved: {report_path}")
        logger.info(f"JSON results: {json_path}")

        return results

    except Exception as e:
        logger.error(f"Full NLI evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
