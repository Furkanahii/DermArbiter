"""DermArbiter Evaluation — MetricsCalculator.

Provides a unified interface for computing all diagnostic evaluation metrics
required across benchmarks (HAM10000, Derm7pt, SkinCon, Fitzpatrick17k):

    • Classification: accuracy, balanced accuracy, top-k accuracy,
      sensitivity, specificity
    • Ranking: F1 (macro / weighted / per-class), AUROC
    • Calibration: ECE (Expected Calibration Error), Brier score
    • Fairness-aware: delta-accuracy across subgroups

All public methods are designed to consume the JSONL record format produced
by ``dermarbiter.experiments.runner.BenchmarkRunner``.

Usage::

    from dermarbiter.evaluation.metrics import MetricsCalculator

    calc = MetricsCalculator.from_jsonl("results/ham10000.jsonl")
    report = calc.compute_all()
    calc.print_report()
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* when the denominator is zero."""
    return numerator / denominator if denominator > 0 else default


# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """Compute comprehensive evaluation metrics for DermArbiter results.

    The calculator operates on a list of *record dicts* that must contain at
    least the following keys:

    * ``predicted``         – top-1 predicted label (str)
    * ``ground_truth``      – ground-truth label (str)

    Optional keys used when available:

    * ``final_diagnosis``   – ranked list of predicted labels (for top-k)
    * ``consensus_score``   – model confidence in [0, 1] (for calibration)
    * ``probabilities``     – dict mapping label → probability (for AUROC)
    * ``fitzpatrick_type``  – Fitzpatrick skin type I–VI (for fairness)

    Parameters
    ----------
    records : list of dict, optional
        Pre-loaded result records.  Alternatively use :meth:`from_jsonl`.
    class_names : list of str, optional
        Fixed set of class labels.  Inferred from data when omitted.
    """

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self._records: List[Dict[str, Any]] = list(records) if records else []
        self._class_names = class_names
        # Lazily materialised
        self._y_true: Optional[List[str]] = None
        self._y_pred: Optional[List[str]] = None

    # ----- Factory ---------------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: str | Path, class_names: Optional[List[str]] = None) -> "MetricsCalculator":
        """Create a calculator from a JSONL results file."""
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON on line %d: %s", line_num, exc)
        logger.info("Loaded %d records from %s", len(records), path)
        return cls(records=records, class_names=class_names)

    # ----- Properties ------------------------------------------------------

    @property
    def records(self) -> List[Dict[str, Any]]:
        return self._records

    @property
    def n_cases(self) -> int:
        return len(self._records)

    @property
    def class_names(self) -> List[str]:
        if self._class_names is not None:
            return self._class_names
        labels: set[str] = set()
        for r in self._records:
            gt = r.get("ground_truth", "").strip().lower()
            if gt:
                labels.add(gt)
        return sorted(labels)

    # ----- Internal accessors ----------------------------------------------

    def _labels(self) -> Tuple[List[str], List[str]]:
        """Return (y_true, y_pred) as lowercased stripped strings."""
        if self._y_true is None:
            self._y_true = [r.get("ground_truth", "").strip().lower() for r in self._records]
            self._y_pred = [r.get("predicted", "").strip().lower() for r in self._records]
        return self._y_true, self._y_pred  # type: ignore[return-value]

    # =====================================================================
    # Classification metrics
    # =====================================================================

    def accuracy(self) -> float:
        """Top-1 accuracy."""
        if not self._records:
            return 0.0
        y_true, y_pred = self._labels()
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return _safe_divide(correct, len(y_true))

    def balanced_accuracy(self) -> float:
        """Balanced accuracy: macro-average of per-class recall."""
        if not self._records:
            return 0.0
        y_true, y_pred = self._labels()
        recalls = []
        for cls_name in self.class_names:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p == cls_name)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p != cls_name)
            recalls.append(_safe_divide(tp, tp + fn))
        return _safe_divide(sum(recalls), len(recalls)) if recalls else 0.0

    def topk_accuracy(self, k: int = 3) -> float:
        """Top-k accuracy: ground truth appears in the first *k* predictions.

        Uses ``final_diagnosis`` list when available; falls back to top-1.
        """
        if not self._records:
            return 0.0
        hits = 0
        for r in self._records:
            gt = r.get("ground_truth", "").strip().lower()
            dx_list = [d.strip().lower() for d in r.get("final_diagnosis", [])][:k]
            # Fallback: if final_diagnosis is empty, use predicted
            if not dx_list:
                dx_list = [r.get("predicted", "").strip().lower()]
            if gt in dx_list:
                hits += 1
        return _safe_divide(hits, len(self._records))

    def sensitivity(self, positive_label: Optional[str] = None) -> Dict[str, float]:
        """Per-class sensitivity (recall / true-positive rate).

        If *positive_label* is given, returns only that class.
        """
        y_true, y_pred = self._labels()
        targets = [positive_label.lower()] if positive_label else self.class_names
        result: Dict[str, float] = {}
        for cls_name in targets:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p == cls_name)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p != cls_name)
            result[cls_name] = _safe_divide(tp, tp + fn)
        return result

    def specificity(self, positive_label: Optional[str] = None) -> Dict[str, float]:
        """Per-class specificity (true-negative rate)."""
        y_true, y_pred = self._labels()
        targets = [positive_label.lower()] if positive_label else self.class_names
        result: Dict[str, float] = {}
        for cls_name in targets:
            tn = sum(1 for t, p in zip(y_true, y_pred) if t != cls_name and p != cls_name)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_name and p == cls_name)
            result[cls_name] = _safe_divide(tn, tn + fp)
        return result

    # =====================================================================
    # F1 metrics
    # =====================================================================

    def per_class_f1(self) -> Dict[str, float]:
        """Per-class F1 scores."""
        y_true, y_pred = self._labels()
        result: Dict[str, float] = {}
        for cls_name in self.class_names:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p == cls_name)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_name and p == cls_name)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_name and p != cls_name)
            precision = _safe_divide(tp, tp + fp)
            recall = _safe_divide(tp, tp + fn)
            result[cls_name] = _safe_divide(2 * precision * recall, precision + recall)
        return result

    def macro_f1(self) -> float:
        """Macro-averaged F1."""
        f1s = self.per_class_f1()
        return _safe_divide(sum(f1s.values()), len(f1s)) if f1s else 0.0

    def weighted_f1(self) -> float:
        """Support-weighted F1."""
        f1s = self.per_class_f1()
        if not f1s:
            return 0.0
        y_true, _ = self._labels()
        support = Counter(y_true)
        total = sum(support.values())
        if total == 0:
            return 0.0
        return sum(f1s.get(label, 0.0) * support.get(label, 0) for label in f1s) / total

    # =====================================================================
    # AUROC
    # =====================================================================

    def auroc(self, average: str = "macro") -> float:
        """Area Under the ROC Curve.

        Requires ``probabilities`` dict in each record (label → prob).
        Falls back to consensus_score as a binary confidence if probabilities
        are unavailable.

        Parameters
        ----------
        average : str
            ``"macro"`` (default) or ``"weighted"``.
        """
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import label_binarize
        except ImportError:
            logger.warning("scikit-learn required for AUROC; returning 0.0")
            return 0.0

        y_true, _ = self._labels()
        classes = self.class_names
        if len(classes) < 2:
            return 0.0

        # Build probability matrix (n_samples × n_classes)
        y_score: List[List[float]] = []
        has_probs = False
        for r in self._records:
            probs = r.get("probabilities", {})
            if probs:
                has_probs = True
                row = [float(probs.get(c, 0.0)) for c in classes]
            else:
                # Fallback: place consensus_score on predicted class
                pred = r.get("predicted", "").strip().lower()
                conf = float(r.get("consensus_score", 0.5))
                row = []
                for c in classes:
                    if c == pred:
                        row.append(conf)
                    else:
                        row.append((1.0 - conf) / max(len(classes) - 1, 1))
            y_score.append(row)

        y_true_bin = label_binarize(y_true, classes=classes)
        y_score_arr = np.array(y_score)

        # Guard against classes with no positive samples
        try:
            return float(roc_auc_score(
                y_true_bin,
                y_score_arr,
                multi_class="ovr",
                average=average,
            ))
        except ValueError as exc:
            logger.warning("AUROC computation failed: %s", exc)
            return 0.0

    # =====================================================================
    # Calibration metrics
    # =====================================================================

    def ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error with equal-width binning.

        Uses ``consensus_score`` as the confidence proxy.
        """
        if not self._records:
            return 0.0

        confidences = np.array([r.get("consensus_score", 0.0) for r in self._records])
        y_true, y_pred = self._labels()
        accuracies = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)])

        n = len(confidences)
        ece_val = 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i < n_bins - 1:
                mask = (confidences >= lo) & (confidences < hi)
            else:
                mask = (confidences >= lo) & (confidences <= hi)
            bin_size = mask.sum()
            if bin_size == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece_val += (bin_size / n) * abs(bin_acc - bin_conf)

        return float(ece_val)

    def brier_score(self) -> float:
        """Brier score: mean (confidence − correct)².

        Uses ``consensus_score`` as the confidence proxy.
        """
        if not self._records:
            return 0.0
        y_true, y_pred = self._labels()
        total = 0.0
        for r, t, p in zip(self._records, y_true, y_pred):
            conf = float(r.get("consensus_score", 0.0))
            correct = 1.0 if t == p else 0.0
            total += (conf - correct) ** 2
        return total / len(self._records)

    # =====================================================================
    # Delta-accuracy (fairness)
    # =====================================================================

    def delta_accuracy(self, group_key: str = "fitzpatrick_type") -> Dict[str, Any]:
        """Accuracy gap across demographic subgroups.

        Computes per-group accuracy and reports:
        - ``per_group``: dict of group → accuracy
        - ``max_delta``: maximum accuracy difference between any two groups
        - ``min_group`` / ``max_group``: groups with min / max accuracy

        Parameters
        ----------
        group_key : str
            Record key containing the demographic group label.
        """
        groups: Dict[str, List[bool]] = defaultdict(list)
        y_true, y_pred = self._labels()
        for r, t, p in zip(self._records, y_true, y_pred):
            g = str(r.get(group_key, "unknown")).strip()
            groups[g].append(t == p)

        per_group = {g: _safe_divide(sum(v), len(v)) for g, v in groups.items()}
        if len(per_group) < 2:
            return {
                "per_group": per_group,
                "max_delta": 0.0,
                "min_group": "",
                "max_group": "",
            }

        min_g = min(per_group, key=per_group.get)  # type: ignore[arg-type]
        max_g = max(per_group, key=per_group.get)  # type: ignore[arg-type]
        return {
            "per_group": per_group,
            "max_delta": per_group[max_g] - per_group[min_g],
            "min_group": min_g,
            "max_group": max_g,
        }

    # =====================================================================
    # Confusion matrix
    # =====================================================================

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Return nested dict ``{true_label: {predicted_label: count}}``."""
        y_true, y_pred = self._labels()
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
        return {k: dict(v) for k, v in matrix.items()}

    # =====================================================================
    # Efficiency metrics
    # =====================================================================

    def early_exit_rate(self) -> float:
        """Fraction of cases that triggered early exit."""
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if r.get("early_exit")) / len(self._records)

    def avg_debate_rounds(self) -> float:
        """Average number of debate rounds."""
        if not self._records:
            return 0.0
        return sum(r.get("num_debate_rounds", 0) for r in self._records) / len(self._records)

    def avg_tokens(self) -> float:
        """Average total tokens per case."""
        if not self._records:
            return 0.0
        return sum(r.get("total_tokens", 0) for r in self._records) / len(self._records)

    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self._records:
            return 0.0
        return sum(r.get("latency_ms", 0.0) for r in self._records) / len(self._records)

    def avg_tool_calls(self) -> float:
        """Average tool calls per case."""
        if not self._records:
            return 0.0
        return sum(r.get("total_tool_calls", 0) for r in self._records) / len(self._records)

    # =====================================================================
    # Aggregation
    # =====================================================================

    def compute_all(self, include_fairness: bool = False) -> Dict[str, Any]:
        """Compute all metrics and return as a flat dictionary.

        Parameters
        ----------
        include_fairness : bool
            If True, also compute ``delta_accuracy`` across Fitzpatrick types.
        """
        result: Dict[str, Any] = {
            "n_cases": self.n_cases,
            # Classification
            "accuracy": self.accuracy(),
            "balanced_accuracy": self.balanced_accuracy(),
            "top3_accuracy": self.topk_accuracy(k=3),
            # F1
            "per_class_f1": self.per_class_f1(),
            "macro_f1": self.macro_f1(),
            "weighted_f1": self.weighted_f1(),
            # AUROC
            "auroc": self.auroc(),
            # Sensitivity / specificity
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
            # Calibration
            "ece": self.ece(),
            "brier_score": self.brier_score(),
            # Confusion
            "confusion_matrix": self.confusion_matrix(),
            # Efficiency
            "early_exit_rate": self.early_exit_rate(),
            "avg_debate_rounds": self.avg_debate_rounds(),
            "avg_tokens": self.avg_tokens(),
            "avg_latency_ms": self.avg_latency_ms(),
            "avg_tool_calls": self.avg_tool_calls(),
        }
        if include_fairness:
            result["delta_accuracy"] = self.delta_accuracy()
        return result

    # =====================================================================
    # Reporting
    # =====================================================================

    def print_report(self, include_fairness: bool = False) -> None:
        """Print a formatted evaluation report to stdout."""
        d = self.compute_all(include_fairness=include_fairness)
        sep = "═" * 64
        print(f"\n{sep}")
        print("  DermArbiter Evaluation Report")
        print(sep)
        print(f"  Cases evaluated:         {d['n_cases']}")
        print()
        print("  ── Classification ──────────────────────────────────")
        print(f"  Top-1 Accuracy:          {d['accuracy']:.4f}")
        print(f"  Balanced Accuracy:       {d['balanced_accuracy']:.4f}")
        print(f"  Top-3 Accuracy:          {d['top3_accuracy']:.4f}")
        print()
        print("  ── F1 Scores ───────────────────────────────────────")
        print(f"  Macro-F1:                {d['macro_f1']:.4f}")
        print(f"  Weighted-F1:             {d['weighted_f1']:.4f}")
        print()
        print("  ── Ranking ─────────────────────────────────────────")
        print(f"  AUROC (macro):           {d['auroc']:.4f}")
        print()
        print("  ── Calibration ─────────────────────────────────────")
        print(f"  ECE:                     {d['ece']:.4f}")
        print(f"  Brier Score:             {d['brier_score']:.4f}")
        print()
        print("  ── Efficiency ──────────────────────────────────────")
        print(f"  Early Exit Rate:         {d['early_exit_rate']:.4f}")
        print(f"  Avg Debate Rounds:       {d['avg_debate_rounds']:.2f}")
        print(f"  Avg Tokens:              {d['avg_tokens']:.0f}")
        print(f"  Avg Tool Calls:          {d['avg_tool_calls']:.1f}")
        print(f"  Avg Latency (ms):        {d['avg_latency_ms']:.1f}")

        # Per-class F1
        f1s = d.get("per_class_f1", {})
        if f1s:
            print()
            print("  ── Per-Class F1 ────────────────────────────────────")
            for label in sorted(f1s):
                sens = d["sensitivity"].get(label, 0.0)
                spec = d["specificity"].get(label, 0.0)
                print(f"    {label:30s}  F1={f1s[label]:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

        # Fairness
        if include_fairness and "delta_accuracy" in d:
            da = d["delta_accuracy"]
            print()
            print("  ── Fairness (Fitzpatrick) ──────────────────────────")
            for g, acc in sorted(da.get("per_group", {}).items()):
                print(f"    Type {g:5s}  Accuracy={acc:.4f}")
            print(f"    Max Δ-Accuracy:        {da['max_delta']:.4f}")
            print(f"    Best group:            {da['max_group']}")
            print(f"    Worst group:           {da['min_group']}")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DermArbiter Metrics Calculator")
    parser.add_argument("--results", required=True, help="Path to JSONL results file.")
    parser.add_argument("--fairness", action="store_true", help="Include fairness metrics.")
    parser.add_argument("--output-json", default=None, help="Export metrics to JSON file.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    calc = MetricsCalculator.from_jsonl(args.results)
    calc.print_report(include_fairness=args.fairness)

    if args.output_json:
        report = calc.compute_all(include_fairness=args.fairness)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Metrics exported to %s", args.output_json)


if __name__ == "__main__":
    main()
