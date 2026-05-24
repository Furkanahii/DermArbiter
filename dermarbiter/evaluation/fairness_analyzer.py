"""DermArbiter Evaluation — FairnessAnalyzer.

Provides systematic fairness evaluation of DermArbiter predictions across
Fitzpatrick skin type subgroups (I–VI).  Implements:

    • Per-group classification metrics (accuracy, sensitivity, specificity, F1)
    • Equalized Odds: |TPR_a − TPR_b| and |FPR_a − FPR_b| across groups
    • Demographic Parity: |P(Ŷ=1|A=a) − P(Ŷ=1|A=b)| across groups
    • Calibration gap: per-group ECE comparison
    • Disparity summary with clinical interpretation

Designed around the Fitzpatrick17k benchmark but can be used with any
dataset that includes demographic group annotations.

Usage::

    from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer

    analyzer = FairnessAnalyzer.from_jsonl(
        "results/fitzpatrick17k.jsonl",
        group_key="fitzpatrick_type",
    )
    report = analyzer.compute_all()
    analyzer.print_report()
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def _ece_from_arrays(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width-bin ECE."""
    n = len(confidences)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 \
            else (confidences >= lo) & (confidences <= hi)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# GroupMetrics — per-subgroup statistics
# ---------------------------------------------------------------------------

class GroupMetrics:
    """Classification metrics for a single demographic group."""

    def __init__(
        self,
        group_name: str,
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float],
    ) -> None:
        self.group_name = group_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.confidences = confidences
        self.n = len(y_true)

    @property
    def accuracy(self) -> float:
        if self.n == 0:
            return 0.0
        return sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p) / self.n

    @property
    def correct_mask(self) -> np.ndarray:
        return np.array([t == p for t, p in zip(self.y_true, self.y_pred)], dtype=float)

    def sensitivity_for_class(self, cls: str) -> float:
        """TPR for a specific class (one-vs-rest)."""
        tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == cls and p == cls)
        fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == cls and p != cls)
        return _safe_divide(tp, tp + fn)

    def specificity_for_class(self, cls: str) -> float:
        """TNR for a specific class (one-vs-rest)."""
        tn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != cls and p != cls)
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != cls and p == cls)
        return _safe_divide(tn, tn + fp)

    def fpr_for_class(self, cls: str) -> float:
        """FPR for a specific class (one-vs-rest)."""
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != cls and p == cls)
        tn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != cls and p != cls)
        return _safe_divide(fp, fp + tn)

    def positive_rate_for_class(self, cls: str) -> float:
        """P(Ŷ = cls) — prediction rate regardless of ground truth."""
        pos = sum(1 for p in self.y_pred if p == cls)
        return _safe_divide(pos, self.n)

    def f1_for_class(self, cls: str) -> float:
        """F1 for a specific class (one-vs-rest)."""
        tp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(self.y_true, self.y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == cls and p != cls)
        prec = _safe_divide(tp, tp + fp)
        rec = _safe_divide(tp, tp + fn)
        return _safe_divide(2 * prec * rec, prec + rec)

    def macro_f1(self) -> float:
        """Macro-averaged F1 across classes present in this group."""
        classes = set(self.y_true)
        if not classes:
            return 0.0
        f1s = [self.f1_for_class(c) for c in classes]
        return sum(f1s) / len(f1s)

    def ece(self, n_bins: int = 10) -> float:
        """ECE for this group."""
        return _ece_from_arrays(
            np.array(self.confidences),
            self.correct_mask,
            n_bins=n_bins,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group_name,
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1(),
            "ece": self.ece(),
        }


# ---------------------------------------------------------------------------
# FairnessAnalyzer
# ---------------------------------------------------------------------------

class FairnessAnalyzer:
    """Evaluate fairness of DermArbiter across demographic subgroups.

    Parameters
    ----------
    records : list of dict
        Result records (from BenchmarkRunner JSONL output).
    group_key : str
        Key in each record identifying the demographic group
        (e.g. ``"fitzpatrick_type"``).
    """

    FITZPATRICK_ORDER = ["I", "II", "III", "IV", "V", "VI"]

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        group_key: str = "fitzpatrick_type",
    ) -> None:
        self._records: List[Dict[str, Any]] = list(records) if records else []
        self.group_key = group_key
        self._groups: Optional[Dict[str, GroupMetrics]] = None

    # ----- Factory ---------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        group_key: str = "fitzpatrick_type",
    ) -> "FairnessAnalyzer":
        """Create from a JSONL results file."""
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return cls(records=records, group_key=group_key)

    # ----- Grouping --------------------------------------------------------

    def _build_groups(self) -> Dict[str, GroupMetrics]:
        """Partition records by group and compute per-group metrics."""
        if self._groups is not None:
            return self._groups

        buckets: Dict[str, Dict[str, list]] = defaultdict(
            lambda: {"y_true": [], "y_pred": [], "confs": []}
        )

        for r in self._records:
            group = str(r.get(self.group_key, "unknown")).strip()
            gt = r.get("ground_truth", "").strip().lower()
            pred = r.get("predicted", "").strip().lower()
            conf = float(r.get("consensus_score", 0.0))
            buckets[group]["y_true"].append(gt)
            buckets[group]["y_pred"].append(pred)
            buckets[group]["confs"].append(conf)

        self._groups = {
            g: GroupMetrics(g, d["y_true"], d["y_pred"], d["confs"])
            for g, d in buckets.items()
        }
        return self._groups

    @property
    def group_names(self) -> List[str]:
        """Sorted list of group names."""
        groups = self._build_groups()
        # Try Fitzpatrick order, else alphabetical
        known = [g for g in self.FITZPATRICK_ORDER if g in groups]
        unknown = sorted(g for g in groups if g not in self.FITZPATRICK_ORDER)
        return known + unknown

    # =====================================================================
    # Per-group metrics
    # =====================================================================

    def per_group_accuracy(self) -> Dict[str, float]:
        groups = self._build_groups()
        return {g: groups[g].accuracy for g in self.group_names}

    def per_group_f1(self) -> Dict[str, float]:
        groups = self._build_groups()
        return {g: groups[g].macro_f1() for g in self.group_names}

    def per_group_ece(self) -> Dict[str, float]:
        groups = self._build_groups()
        return {g: groups[g].ece() for g in self.group_names}

    def per_group_n(self) -> Dict[str, int]:
        groups = self._build_groups()
        return {g: groups[g].n for g in self.group_names}

    # =====================================================================
    # Equalized Odds
    # =====================================================================

    def equalized_odds(
        self,
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute equalized odds across demographic groups.

        Equalized odds requires equal TPR and FPR across groups for each
        class.  We report the maximum disparity across all group pairs.

        Parameters
        ----------
        target_classes : list of str, optional
            Classes to evaluate.  Defaults to all classes found in data.

        Returns
        -------
        dict with keys:
            per_class : dict mapping class → {tpr_disparity, fpr_disparity, worst_pair}
            max_tpr_disparity : float
            max_fpr_disparity : float
            satisfied : bool (max disparities ≤ 0.05)
        """
        groups = self._build_groups()
        names = self.group_names
        if len(names) < 2:
            return {
                "per_class": {},
                "max_tpr_disparity": 0.0,
                "max_fpr_disparity": 0.0,
                "satisfied": True,
            }

        # Determine target classes
        if target_classes is None:
            all_classes: Set[str] = set()
            for g in groups.values():
                all_classes.update(g.y_true)
            target_classes = sorted(all_classes)

        per_class: Dict[str, Dict[str, Any]] = {}
        global_max_tpr = 0.0
        global_max_fpr = 0.0

        for cls in target_classes:
            tprs = {g: groups[g].sensitivity_for_class(cls) for g in names}
            fprs = {g: groups[g].fpr_for_class(cls) for g in names}

            max_tpr_disp = 0.0
            max_fpr_disp = 0.0
            worst_pair_tpr = ("", "")
            worst_pair_fpr = ("", "")

            for ga, gb in combinations(names, 2):
                tpr_d = abs(tprs[ga] - tprs[gb])
                fpr_d = abs(fprs[ga] - fprs[gb])
                if tpr_d > max_tpr_disp:
                    max_tpr_disp = tpr_d
                    worst_pair_tpr = (ga, gb)
                if fpr_d > max_fpr_disp:
                    max_fpr_disp = fpr_d
                    worst_pair_fpr = (ga, gb)

            per_class[cls] = {
                "tpr_disparity": max_tpr_disp,
                "fpr_disparity": max_fpr_disp,
                "tpr_per_group": tprs,
                "fpr_per_group": fprs,
                "worst_pair_tpr": worst_pair_tpr,
                "worst_pair_fpr": worst_pair_fpr,
            }
            global_max_tpr = max(global_max_tpr, max_tpr_disp)
            global_max_fpr = max(global_max_fpr, max_fpr_disp)

        return {
            "per_class": per_class,
            "max_tpr_disparity": global_max_tpr,
            "max_fpr_disparity": global_max_fpr,
            "satisfied": global_max_tpr <= 0.05 and global_max_fpr <= 0.05,
        }

    # =====================================================================
    # Demographic Parity
    # =====================================================================

    def demographic_parity(
        self,
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute demographic parity across groups.

        Demographic parity requires P(Ŷ=c | A=a) = P(Ŷ=c | A=b) for all
        groups a, b and class c.

        Returns
        -------
        dict with keys:
            per_class : dict mapping class → {disparity, per_group_rate, worst_pair}
            max_disparity : float
            satisfied : bool (max disparity ≤ 0.05)
        """
        groups = self._build_groups()
        names = self.group_names
        if len(names) < 2:
            return {
                "per_class": {},
                "max_disparity": 0.0,
                "satisfied": True,
            }

        if target_classes is None:
            all_classes: Set[str] = set()
            for g in groups.values():
                all_classes.update(set(g.y_pred))
            target_classes = sorted(all_classes)

        per_class: Dict[str, Dict[str, Any]] = {}
        global_max = 0.0

        for cls in target_classes:
            rates = {g: groups[g].positive_rate_for_class(cls) for g in names}

            max_disp = 0.0
            worst_pair = ("", "")
            for ga, gb in combinations(names, 2):
                d = abs(rates[ga] - rates[gb])
                if d > max_disp:
                    max_disp = d
                    worst_pair = (ga, gb)

            per_class[cls] = {
                "disparity": max_disp,
                "per_group_rate": rates,
                "worst_pair": worst_pair,
            }
            global_max = max(global_max, max_disp)

        return {
            "per_class": per_class,
            "max_disparity": global_max,
            "satisfied": global_max <= 0.05,
        }

    # =====================================================================
    # Calibration gap
    # =====================================================================

    def calibration_gap(self) -> Dict[str, Any]:
        """Compute per-group ECE and the maximum ECE gap.

        Returns
        -------
        dict with keys:
            per_group_ece : dict mapping group → ECE
            max_gap : float
            worst_pair : tuple of (group_a, group_b)
        """
        eces = self.per_group_ece()
        if len(eces) < 2:
            return {"per_group_ece": eces, "max_gap": 0.0, "worst_pair": ("", "")}

        max_gap = 0.0
        worst = ("", "")
        for ga, gb in combinations(eces, 2):
            gap = abs(eces[ga] - eces[gb])
            if gap > max_gap:
                max_gap = gap
                worst = (ga, gb)

        return {
            "per_group_ece": eces,
            "max_gap": max_gap,
            "worst_pair": worst,
        }

    # =====================================================================
    # Accuracy gap (delta-accuracy)
    # =====================================================================

    def accuracy_gap(self) -> Dict[str, Any]:
        """Max accuracy difference across groups."""
        accs = self.per_group_accuracy()
        if len(accs) < 2:
            return {"per_group": accs, "max_delta": 0.0, "min_group": "", "max_group": ""}

        min_g = min(accs, key=accs.get)  # type: ignore[arg-type]
        max_g = max(accs, key=accs.get)  # type: ignore[arg-type]
        return {
            "per_group": accs,
            "max_delta": accs[max_g] - accs[min_g],
            "min_group": min_g,
            "max_group": max_g,
        }

    # =====================================================================
    # Aggregation
    # =====================================================================

    def compute_all(
        self,
        target_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute all fairness metrics.

        Returns
        -------
        dict
            Full fairness report with per-group metrics and disparity measures.
        """
        return {
            "n_total": len(self._records),
            "n_groups": len(self.group_names),
            "group_names": self.group_names,
            "per_group_n": self.per_group_n(),
            "per_group_accuracy": self.per_group_accuracy(),
            "per_group_f1": self.per_group_f1(),
            "accuracy_gap": self.accuracy_gap(),
            "equalized_odds": self.equalized_odds(target_classes),
            "demographic_parity": self.demographic_parity(target_classes),
            "calibration_gap": self.calibration_gap(),
        }

    # =====================================================================
    # Reporting
    # =====================================================================

    def print_report(
        self,
        target_classes: Optional[List[str]] = None,
    ) -> None:
        """Print a formatted fairness evaluation report."""
        report = self.compute_all(target_classes)
        sep = "═" * 64

        print(f"\n{sep}")
        print("  DermArbiter Fairness Report")
        print(sep)
        print(f"  Total cases:           {report['n_total']}")
        print(f"  Groups evaluated:      {report['n_groups']}")
        print(f"  Group key:             {self.group_key}")

        # Per-group breakdown
        print()
        print("  ── Per-Group Performance ───────────────────────────")
        print(f"  {'Group':8s}  {'N':>6s}  {'Acc':>7s}  {'F1':>7s}  {'ECE':>7s}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")

        accs = report["per_group_accuracy"]
        f1s = report["per_group_f1"]
        eces = report["calibration_gap"]["per_group_ece"]
        ns = report["per_group_n"]

        for g in self.group_names:
            print(
                f"  {g:8s}  {ns.get(g, 0):6d}  "
                f"{accs.get(g, 0.0):7.4f}  "
                f"{f1s.get(g, 0.0):7.4f}  "
                f"{eces.get(g, 0.0):7.4f}"
            )

        # Accuracy gap
        ag = report["accuracy_gap"]
        print()
        print("  ── Accuracy Gap ────────────────────────────────────")
        print(f"  Max Δ-Accuracy:        {ag['max_delta']:.4f}")
        print(f"  Best group:            {ag['max_group']}")
        print(f"  Worst group:           {ag['min_group']}")

        # Equalized odds
        eo = report["equalized_odds"]
        print()
        print("  ── Equalized Odds ──────────────────────────────────")
        print(f"  Max TPR disparity:     {eo['max_tpr_disparity']:.4f}")
        print(f"  Max FPR disparity:     {eo['max_fpr_disparity']:.4f}")
        satisfied_str = "✓ YES" if eo["satisfied"] else "✗ NO"
        print(f"  Satisfied (≤ 0.05):    {satisfied_str}")

        # Demographic parity
        dp = report["demographic_parity"]
        print()
        print("  ── Demographic Parity ──────────────────────────────")
        print(f"  Max disparity:         {dp['max_disparity']:.4f}")
        satisfied_str = "✓ YES" if dp["satisfied"] else "✗ NO"
        print(f"  Satisfied (≤ 0.05):    {satisfied_str}")

        # Calibration gap
        cg = report["calibration_gap"]
        print()
        print("  ── Calibration Gap ─────────────────────────────────")
        print(f"  Max ECE gap:           {cg['max_gap']:.4f}")
        if cg["worst_pair"] != ("", ""):
            print(f"  Worst pair:            {cg['worst_pair'][0]} vs {cg['worst_pair'][1]}")

        # Clinical interpretation
        print()
        print("  ── Clinical Interpretation ─────────────────────────")
        issues = []
        if ag["max_delta"] > 0.10:
            issues.append(
                f"  ⚠ Large accuracy gap ({ag['max_delta']:.2%}) between "
                f"{ag['min_group']} and {ag['max_group']}. "
                f"Model may underperform on darker skin types."
            )
        if not eo["satisfied"]:
            issues.append(
                "  ⚠ Equalized odds NOT satisfied. TPR/FPR vary across groups."
            )
        if not dp["satisfied"]:
            issues.append(
                "  ⚠ Demographic parity NOT satisfied. Prediction rates vary across groups."
            )
        if cg["max_gap"] > 0.05:
            issues.append(
                f"  ⚠ Calibration inconsistent across groups (gap={cg['max_gap']:.4f})."
            )

        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  ✓ No significant fairness concerns detected.")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DermArbiter Fairness Analyzer")
    parser.add_argument("--results", required=True, help="Path to JSONL results file.")
    parser.add_argument(
        "--group-key", default="fitzpatrick_type",
        help="Record key for demographic group (default: fitzpatrick_type).",
    )
    parser.add_argument("--output-json", default=None, help="Export fairness report to JSON.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analyzer = FairnessAnalyzer.from_jsonl(args.results, group_key=args.group_key)
    analyzer.print_report()

    if args.output_json:
        report = analyzer.compute_all()
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Fairness report exported to %s", args.output_json)


if __name__ == "__main__":
    main()
