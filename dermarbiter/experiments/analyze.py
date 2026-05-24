"""DermArbiter Results Analyzer — Metrics & Reporting for Benchmark Runs.

Reads JSONL results produced by ``BenchmarkRunner`` and computes:
    • Accuracy, Top-3 accuracy
    • Per-class F1, Macro-F1, Weighted-F1
    • Calibration: ECE (Expected Calibration Error), Brier score
    • Efficiency: early-exit rate, avg debate rounds, avg tokens, avg latency
    • Confusion matrix (as dict)

Usage::

    python -m dermarbiter.experiments.analyze \
        --results results/run_001.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python metrics (no sklearn / numpy dependency)
# ---------------------------------------------------------------------------

def _compute_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_ece(
    confidences: List[float],
    accuracies: List[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error with equal-width bins.

    Args:
        confidences: Per-sample confidence scores ∈ [0, 1].
        accuracies:  Per-sample correctness flags.
        n_bins:      Number of bins.

    Returns:
        ECE ∈ [0, 1].
    """
    if not confidences:
        return 0.0

    n = len(confidences)
    ece = 0.0
    bin_width = 1.0 / n_bins

    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width
        # Samples falling in this bin
        indices = [
            i for i, c in enumerate(confidences)
            if lo <= c < hi or (b == n_bins - 1 and c == hi)
        ]
        if not indices:
            continue
        bin_acc = sum(1 for i in indices if accuracies[i]) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_acc - bin_conf)

    return ece


def _compute_brier(
    confidences: List[float],
    accuracies: List[bool],
) -> float:
    """Compute Brier score: mean(confidence - correct)^2."""
    if not confidences:
        return 0.0
    total = sum(
        (conf - (1.0 if acc else 0.0)) ** 2
        for conf, acc in zip(confidences, accuracies)
    )
    return total / len(confidences)


# ---------------------------------------------------------------------------
# ResultsAnalyzer
# ---------------------------------------------------------------------------

class ResultsAnalyzer:
    """Compute comprehensive metrics from a benchmark results JSONL file.

    Args:
        results_path: Path to the JSONL file written by ``BenchmarkRunner``.
    """

    def __init__(self, results_path: Optional[str] = None) -> None:
        self.results_path = results_path
        self._records: List[Dict[str, Any]] = []
        if results_path is not None:
            self.load(results_path)

    # ----- IO --------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load results from a JSONL file."""
        self._records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        logger.info("Loaded %d result records from %s", len(self._records), path)

    def load_records(self, records: List[Dict[str, Any]]) -> None:
        """Load results from an in-memory list (useful for testing)."""
        self._records = list(records)

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records

    # ----- Core metrics ----------------------------------------------------

    def accuracy(self) -> float:
        """Top-1 accuracy: predicted == ground_truth (case-insensitive)."""
        if not self._records:
            return 0.0
        correct = sum(
            1 for r in self._records
            if r.get("predicted", "").strip().lower()
            == r.get("ground_truth", "").strip().lower()
        )
        return correct / len(self._records)

    def top3_accuracy(self) -> float:
        """Top-3 accuracy: ground_truth appears in final_diagnosis[:3]."""
        if not self._records:
            return 0.0
        hits = 0
        for r in self._records:
            gt = r.get("ground_truth", "").strip().lower()
            dx_list = [d.strip().lower() for d in r.get("final_diagnosis", [])[:3]]
            if gt in dx_list:
                hits += 1
        return hits / len(self._records)

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Return a nested dict ``{true_label: {predicted_label: count}}``."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self._records:
            gt = r.get("ground_truth", "").strip().lower()
            pred = r.get("predicted", "").strip().lower()
            matrix[gt][pred] += 1
        # Convert to plain dict
        return {k: dict(v) for k, v in matrix.items()}

    def per_class_f1(self) -> Dict[str, float]:
        """Per-class F1 scores."""
        labels = set()
        for r in self._records:
            labels.add(r.get("ground_truth", "").strip().lower())
            labels.add(r.get("predicted", "").strip().lower())
        labels.discard("")

        f1s: Dict[str, float] = {}
        for label in sorted(labels):
            tp = fp = fn = 0
            for r in self._records:
                gt = r.get("ground_truth", "").strip().lower()
                pred = r.get("predicted", "").strip().lower()
                if pred == label and gt == label:
                    tp += 1
                elif pred == label and gt != label:
                    fp += 1
                elif pred != label and gt == label:
                    fn += 1
            f1s[label] = _compute_f1(tp, fp, fn)
        return f1s

    def macro_f1(self) -> float:
        """Macro-averaged F1 (unweighted mean of per-class F1)."""
        f1s = self.per_class_f1()
        if not f1s:
            return 0.0
        return sum(f1s.values()) / len(f1s)

    def weighted_f1(self) -> float:
        """Weighted-averaged F1 (weighted by support per class)."""
        f1s = self.per_class_f1()
        if not f1s:
            return 0.0

        # Count true support per class
        support: Dict[str, int] = Counter()
        for r in self._records:
            gt = r.get("ground_truth", "").strip().lower()
            support[gt] += 1

        total = sum(support.values())
        if total == 0:
            return 0.0

        weighted = sum(f1s.get(label, 0.0) * support.get(label, 0) for label in f1s)
        return weighted / total

    # ----- Calibration -----------------------------------------------------

    def ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error using consensus_score as confidence."""
        confidences = [r.get("consensus_score", 0.0) for r in self._records]
        accuracies = [
            r.get("predicted", "").strip().lower()
            == r.get("ground_truth", "").strip().lower()
            for r in self._records
        ]
        return _compute_ece(confidences, accuracies, n_bins=n_bins)

    def brier_score(self) -> float:
        """Brier score using consensus_score as confidence."""
        confidences = [r.get("consensus_score", 0.0) for r in self._records]
        accuracies = [
            r.get("predicted", "").strip().lower()
            == r.get("ground_truth", "").strip().lower()
            for r in self._records
        ]
        return _compute_brier(confidences, accuracies)

    # ----- Efficiency ------------------------------------------------------

    def early_exit_rate(self) -> float:
        """Fraction of cases that triggered early exit."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.get("early_exit")) / len(self._records)

    def avg_debate_rounds(self) -> float:
        """Average number of debate rounds across cases."""
        if not self._records:
            return 0.0
        return sum(r.get("num_debate_rounds", 0) for r in self._records) / len(self._records)

    def avg_tokens(self) -> float:
        """Average total token usage across cases."""
        if not self._records:
            return 0.0
        return sum(r.get("total_tokens", 0) for r in self._records) / len(self._records)

    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds across cases."""
        if not self._records:
            return 0.0
        return sum(r.get("latency_ms", 0.0) for r in self._records) / len(self._records)

    # ----- Aggregation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return all metrics as a flat dictionary for programmatic access."""
        return {
            "n_cases": len(self._records),
            "accuracy": self.accuracy(),
            "top3_accuracy": self.top3_accuracy(),
            "per_class_f1": self.per_class_f1(),
            "macro_f1": self.macro_f1(),
            "weighted_f1": self.weighted_f1(),
            "ece": self.ece(),
            "brier_score": self.brier_score(),
            "confusion_matrix": self.confusion_matrix(),
            "early_exit_rate": self.early_exit_rate(),
            "avg_debate_rounds": self.avg_debate_rounds(),
            "avg_tokens": self.avg_tokens(),
            "avg_latency_ms": self.avg_latency_ms(),
        }

    def print_report(self) -> None:
        """Print a formatted summary table to stdout."""
        d = self.to_dict()
        sep = "═" * 60
        print(f"\n{sep}")
        print("  DermArbiter Benchmark Report")
        print(sep)
        print(f"  Cases evaluated:       {d['n_cases']}")
        print(f"  Top-1 Accuracy:        {d['accuracy']:.4f}")
        print(f"  Top-3 Accuracy:        {d['top3_accuracy']:.4f}")
        print(f"  Macro-F1:              {d['macro_f1']:.4f}")
        print(f"  Weighted-F1:           {d['weighted_f1']:.4f}")
        print(f"  ECE:                   {d['ece']:.4f}")
        print(f"  Brier Score:           {d['brier_score']:.4f}")
        print(f"  Early Exit Rate:       {d['early_exit_rate']:.4f}")
        print(f"  Avg Debate Rounds:     {d['avg_debate_rounds']:.2f}")
        print(f"  Avg Tokens:            {d['avg_tokens']:.0f}")
        print(f"  Avg Latency (ms):      {d['avg_latency_ms']:.1f}")

        # Per-class F1 breakdown
        f1s = d["per_class_f1"]
        if f1s:
            print(f"\n  {'Per-Class F1':─<40}")
            for label, score in sorted(f1s.items()):
                print(f"    {label:30s} {score:.4f}")

        # Confusion matrix
        cm = d["confusion_matrix"]
        if cm:
            print(f"\n  {'Confusion Matrix':─<40}")
            for true_label, preds in sorted(cm.items()):
                for pred_label, count in sorted(preds.items()):
                    print(f"    true={true_label:20s} → pred={pred_label:20s} : {count}")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter Results Analyzer",
    )
    parser.add_argument(
        "--results", required=True, help="Path to JSONL results file."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = ResultsAnalyzer(results_path=args.results)
    analyzer.print_report()


if __name__ == "__main__":
    main()
