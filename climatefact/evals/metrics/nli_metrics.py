"""
NLI (Natural Language Inference) evaluation metrics for assessing quality of entailment classification.
Implements accuracy, precision, recall, F1-score, and confusion matrix analysis.
"""

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class NLIMetrics:
    """
    Comprehensive NLI metrics calculator for evaluating natural language inference systems.

    Supports evaluation of:
    - Classification accuracy
    - Per-class precision, recall, F1-score
    - Macro/micro averages
    - Confusion matrix analysis
    """

    def __init__(self):
        self.metrics_history = []
        self.label_mapping = {"entailment": "ENTAILMENT", "contradiction": "CONTRADICTION", "neutral": "NEUTRAL"}

    def normalize_label(self, label: str) -> str:
        """Normalize label to standard format."""
        if not label:
            raise ValueError("Empty or None label provided - all labels must be non-empty strings")

        label = label.lower().strip()

        if not label:
            raise ValueError("Label is empty after stripping whitespace")

        # Handle variations
        if label in ["entailment", "entail", "supports", "true"]:
            return "ENTAILMENT"
        elif label in ["contradiction", "contradict", "refutes", "false"]:
            return "CONTRADICTION"
        elif label in ["neutral", "neither", "unknown", "unrelated"]:
            return "NEUTRAL"
        else:
            # Fail loudly instead of defaulting
            raise ValueError(
                f"Unknown label '{label}' - expected one of: entailment, contradiction, neutral (or their variations)"
            )

    def calculate_accuracy(self, predicted_labels: list[str], true_labels: list[str]) -> float:
        """
        Calculate overall accuracy: fraction of correct predictions.

        Args:
            predicted_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if len(predicted_labels) != len(true_labels):
            raise ValueError(f"Length mismatch: predicted={len(predicted_labels)}, true={len(true_labels)}")

        if not predicted_labels:
            raise ValueError("Empty label lists provided - cannot calculate accuracy")

        # Normalize labels
        pred_normalized = [self.normalize_label(label) for label in predicted_labels]
        true_normalized = [self.normalize_label(label) for label in true_labels]

        correct = sum(1 for p, t in zip(pred_normalized, true_normalized, strict=False) if p == t)
        accuracy = correct / len(predicted_labels)

        return accuracy

    def calculate_precision_recall_f1(
        self, predicted_labels: list[str], true_labels: list[str], label: str
    ) -> tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score for a specific label.

        Args:
            predicted_labels: List of predicted labels
            true_labels: List of true labels
            label: Label to calculate metrics for

        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if len(predicted_labels) != len(true_labels):
            raise ValueError(f"Length mismatch: predicted={len(predicted_labels)}, true={len(true_labels)}")

        if not predicted_labels:
            raise ValueError("Empty label lists provided - cannot calculate precision/recall/F1")

        # Normalize labels
        pred_normalized = [self.normalize_label(lbl) for lbl in predicted_labels]
        true_normalized = [self.normalize_label(lbl) for lbl in true_labels]
        target_label = self.normalize_label(label)

        # Calculate TP, FP, FN
        tp = sum(
            1 for p, t in zip(pred_normalized, true_normalized, strict=False) if p == target_label and t == target_label
        )
        fp = sum(
            1 for p, t in zip(pred_normalized, true_normalized, strict=False) if p == target_label and t != target_label
        )
        fn = sum(
            1 for p, t in zip(pred_normalized, true_normalized, strict=False) if p != target_label and t == target_label
        )

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1_score

    def calculate_confusion_matrix(
        self, predicted_labels: list[str], true_labels: list[str]
    ) -> dict[str, dict[str, int]]:
        """
        Calculate confusion matrix for all labels.

        Args:
            predicted_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Nested dictionary representing confusion matrix
        """
        if len(predicted_labels) != len(true_labels):
            raise ValueError(f"Length mismatch: predicted={len(predicted_labels)}, true={len(true_labels)}")

        if not predicted_labels:
            raise ValueError("Empty label lists provided - cannot calculate confusion matrix")

        # Normalize labels
        pred_normalized = [self.normalize_label(lbl) for lbl in predicted_labels]
        true_normalized = [self.normalize_label(lbl) for lbl in true_labels]

        all_labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
        confusion_matrix = {true_label: {pred_label: 0 for pred_label in all_labels} for true_label in all_labels}

        for pred, true in zip(pred_normalized, true_normalized, strict=False):
            confusion_matrix[true][pred] += 1

        return confusion_matrix

    def calculate_macro_averages(self, predicted_labels: list[str], true_labels: list[str]) -> dict[str, float]:
        """
        Calculate macro-averaged precision, recall, and F1-score.

        Args:
            predicted_labels: List of predicted labels
            true_labels: List of true labels

        Returns:
            Dictionary with macro averages
        """
        all_labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
        precisions, recalls, f1_scores = [], [], []

        for label in all_labels:
            precision, recall, f1 = self.calculate_precision_recall_f1(predicted_labels, true_labels, label)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        }

    def evaluate_single_prediction(self, predicted_label: str, true_label: str) -> dict[str, Any]:
        """
        Evaluate a single prediction.

        Args:
            predicted_label: Predicted label
            true_label: True label

        Returns:
            Evaluation results for single prediction
        """
        pred_normalized = self.normalize_label(predicted_label)
        true_normalized = self.normalize_label(true_label)

        return {"predicted": pred_normalized, "true": true_normalized, "correct": pred_normalized == true_normalized}

    def evaluate_multiple_predictions(self, predictions: list[tuple[str, str]]) -> dict[str, Any]:
        """
        Evaluate multiple predictions.

        Args:
            predictions: List of (predicted_label, true_label) tuples

        Returns:
            Comprehensive evaluation results
        """
        if not predictions:
            raise ValueError("No predictions provided for evaluation - cannot proceed with empty dataset")

        predicted_labels = [pred for pred, _ in predictions]
        true_labels = [true for _, true in predictions]

        # Calculate overall metrics
        accuracy = self.calculate_accuracy(predicted_labels, true_labels)
        macro_averages = self.calculate_macro_averages(predicted_labels, true_labels)
        confusion_matrix = self.calculate_confusion_matrix(predicted_labels, true_labels)

        # Calculate per-class metrics
        all_labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
        per_class_metrics = {}

        for label in all_labels:
            precision, recall, f1 = self.calculate_precision_recall_f1(predicted_labels, true_labels, label)
            per_class_metrics[label] = {"precision": precision, "recall": recall, "f1_score": f1}

        return {
            "accuracy": accuracy,
            "total_predictions": len(predictions),
            "per_class_metrics": per_class_metrics,
            "macro_averages": macro_averages,
            "confusion_matrix": confusion_matrix,
            "label_distribution": {
                "predicted": dict(Counter([self.normalize_label(p) for p in predicted_labels])),
                "true": dict(Counter([self.normalize_label(t) for t in true_labels])),
            },
        }

    def get_metrics_summary(self, results: dict[str, Any]) -> str:
        """
        Generate a human-readable summary of metrics.

        Args:
            results: Results from evaluate_multiple_predictions

        Returns:
            Formatted summary string
        """
        if not results:
            raise ValueError("No results provided - cannot generate summary for empty results")

        summary = f"""
NLI EVALUATION SUMMARY
=====================
Total Predictions: {results.get("total_predictions", 0)}
Overall Accuracy: {results.get("accuracy", 0.0):.4f}

MACRO AVERAGES
--------------
Precision: {results.get("macro_averages", {}).get("macro_precision", 0.0):.4f}
Recall: {results.get("macro_averages", {}).get("macro_recall", 0.0):.4f}
F1-Score: {results.get("macro_averages", {}).get("macro_f1", 0.0):.4f}

PER-CLASS METRICS
-----------------"""

        per_class = results.get("per_class_metrics", {})
        for label in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]:
            if label in per_class:
                metrics = per_class[label]
                summary += f"""
{label}:
  Precision: {metrics.get("precision", 0.0):.4f}
  Recall: {metrics.get("recall", 0.0):.4f}
  F1-Score: {metrics.get("f1_score", 0.0):.4f}"""

        # Add confusion matrix
        confusion = results.get("confusion_matrix", {})
        if confusion:
            summary += "\n\nCONFUSION MATRIX"
            summary += "\n----------------"
            summary += "\nTrue\\Predicted\tENT\tCON\tNEU"
            for true_label in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]:
                if true_label in confusion:
                    row = confusion[true_label]
                    summary += (
                        f"\n{true_label[:3]}\t\t{row.get('ENTAILMENT', 0)}"
                        f"\t{row.get('CONTRADICTION', 0)}\t{row.get('NEUTRAL', 0)}"
                    )

        return summary
