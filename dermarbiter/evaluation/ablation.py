"""DermArbiter Evaluation — Tool / Agent / Round ablation analyzer.

Consumes the ablation JSONL produced by ``dermarbiter.experiments.ablation``
(or any compatible runner) and computes:

    • per-variant accuracy and balanced accuracy
    • Δacc = baseline_acc − variant_acc  (positive ⇒ variant *hurt* performance)
    • bootstrap 95% CI on Δacc
    • paired-bootstrap p-value vs. the all-tools / all-agents baseline
    • paper-ready Markdown / dict report

Typical usage::

    from dermarbiter.evaluation.ablation import AblationAnalyzer

    analyzer = AblationAnalyzer.from_jsonl("results/ablation_tool.jsonl")
    report = analyzer.compute()
    print(analyzer.to_markdown())

A CLI is also provided::

    python -m dermarbiter.evaluation.ablation \
        --results results/ablation_tool.jsonl \
        --output  results/ablation_tool_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Baseline variant names emitted by experiments/ablation.py
BASELINE_VARIANTS: Tuple[str, ...] = (
    "baseline_all_tools",
    "baseline_all_agents",
)

DEFAULT_BOOTSTRAP_N = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 1729


@dataclass
class VariantStats:
    """Summary stats for a single ablation variant."""

    variant: str
    ablation_type: str
    n: int
    accuracy: float
    balanced_accuracy: float
    mean_latency_ms: float
    mean_rounds: float
    delta_acc: Optional[float] = None  # baseline − variant
    delta_ci_lower: Optional[float] = None
    delta_ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    contribution_label: str = ""  # "helpful" | "neutral" | "harmful"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "ablation_type": self.ablation_type,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "mean_rounds": round(self.mean_rounds, 2),
            "delta_acc": round(self.delta_acc, 4) if self.delta_acc is not None else None,
            "delta_ci_lower": round(self.delta_ci_lower, 4) if self.delta_ci_lower is not None else None,
            "delta_ci_upper": round(self.delta_ci_upper, 4) if self.delta_ci_upper is not None else None,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "contribution_label": self.contribution_label,
        }


def _accuracy(records: Sequence[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    correct = sum(
        1 for r in records
        if str(r.get("predicted", "")).strip().lower()
        == str(r.get("ground_truth", "")).strip().lower()
        and r.get("predicted")
    )
    return correct / len(records)


def _balanced_accuracy(records: Sequence[Dict[str, Any]]) -> float:
    """Macro-recall across present ground-truth classes."""
    per_class: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        gt = str(r.get("ground_truth", "")).strip().lower()
        pred = str(r.get("predicted", "")).strip().lower()
        if not gt:
            continue
        per_class[gt].append(1 if (pred and pred == gt) else 0)
    if not per_class:
        return 0.0
    recalls = [sum(v) / len(v) for v in per_class.values() if v]
    return sum(recalls) / len(recalls) if recalls else 0.0


def _paired_records(
    baseline: Sequence[Dict[str, Any]],
    variant: Sequence[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    """Align two variants by case_id and return per-case correctness vectors."""
    by_case_b = {r.get("case_id"): r for r in baseline if r.get("case_id")}
    by_case_v = {r.get("case_id"): r for r in variant if r.get("case_id")}
    common = sorted(set(by_case_b) & set(by_case_v))
    b_correct: List[int] = []
    v_correct: List[int] = []
    for cid in common:
        b, v = by_case_b[cid], by_case_v[cid]
        gt = str(b.get("ground_truth", "")).strip().lower()
        b_pred = str(b.get("predicted", "")).strip().lower()
        v_pred = str(v.get("predicted", "")).strip().lower()
        b_correct.append(1 if (b_pred and b_pred == gt) else 0)
        v_correct.append(1 if (v_pred and v_pred == gt) else 0)
    return b_correct, v_correct


def _bootstrap_delta_ci(
    b_correct: Sequence[int],
    v_correct: Sequence[int],
    n_boot: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap on Δacc = baseline − variant.

    Returns (point_estimate, ci_low, ci_high, two_sided_p).
    """
    if not b_correct or len(b_correct) != len(v_correct):
        return 0.0, 0.0, 0.0, 1.0

    rng = random.Random(seed)
    n = len(b_correct)
    point = (sum(b_correct) - sum(v_correct)) / n

    deltas: List[float] = []
    for _ in range(n_boot):
        # Sample indices with replacement (paired bootstrap)
        sample_b = 0
        sample_v = 0
        for _i in range(n):
            j = rng.randrange(n)
            sample_b += b_correct[j]
            sample_v += v_correct[j]
        deltas.append((sample_b - sample_v) / n)

    deltas.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    ci_low = deltas[max(0, lo_idx)]
    ci_high = deltas[min(n_boot - 1, hi_idx)]

    # Two-sided p-value: fraction of bootstrap deltas crossing zero
    # against the observed sign. Centered on zero under H0.
    centered = [d - point for d in deltas]
    centered.sort()
    extreme = sum(1 for d in centered if abs(d) >= abs(point))
    p_value = (extreme + 1) / (n_boot + 1)
    return point, ci_low, ci_high, p_value


def _label_contribution(
    delta: float, ci_low: float, ci_high: float
) -> str:
    """Classify a tool's contribution from its Δacc CI.

    A tool is *helpful* if removing it significantly hurts accuracy
    (Δacc > 0 and CI excludes 0). Symmetric definition for *harmful*.
    """
    if ci_low > 0:
        return "helpful"
    if ci_high < 0:
        return "harmful"
    return "neutral"


class AblationAnalyzer:
    """Analyze ablation result JSONL files.

    Parameters
    ----------
    records:
        List of result dicts (one row per case × variant), as emitted by
        ``dermarbiter.experiments.ablation.AblationRunner``.
    bootstrap_n:
        Number of bootstrap replicates for Δacc CIs (default 2000).
    alpha:
        Two-sided significance level for CIs (default 0.05).
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
        alpha: float = DEFAULT_ALPHA,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._records: List[Dict[str, Any]] = list(records)
        self._bootstrap_n = bootstrap_n
        self._alpha = alpha
        self._seed = seed
        self._stats: List[VariantStats] = []

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "AblationAnalyzer":
        """Load records from a JSONL file produced by AblationRunner."""
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return cls(rows, **kwargs)

    # -- Grouping ----------------------------------------------------------

    def _by_variant(self) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self._records:
            variant = r.get("variant")
            if variant:
                groups[variant].append(r)
        return dict(groups)

    def _baseline_for(
        self, groups: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        for name in BASELINE_VARIANTS:
            if name in groups:
                return groups[name]
        # Fall back to any variant whose name starts with "baseline"
        for name in groups:
            if name.startswith("baseline"):
                return groups[name]
        return None

    # -- Public API --------------------------------------------------------

    def compute(self) -> List[VariantStats]:
        """Compute per-variant stats with Δacc, CI, p-value."""
        groups = self._by_variant()
        if not groups:
            logger.warning("No records grouped — empty ablation result?")
            self._stats = []
            return []

        baseline_records = self._baseline_for(groups)
        baseline_name = None
        if baseline_records is not None:
            for name in BASELINE_VARIANTS:
                if name in groups:
                    baseline_name = name
                    break
            if baseline_name is None:
                baseline_name = next(
                    (n for n in groups if n.startswith("baseline")), None
                )

        out: List[VariantStats] = []
        for variant, rows in groups.items():
            ab_type = rows[0].get("ablation_type", "")
            latencies = [
                float(r.get("latency_ms", 0.0)) for r in rows
                if r.get("latency_ms") is not None
            ]
            rounds = [
                float(r.get("num_debate_rounds", 0)) for r in rows
                if r.get("num_debate_rounds") is not None
            ]

            stats = VariantStats(
                variant=variant,
                ablation_type=ab_type,
                n=len(rows),
                accuracy=_accuracy(rows),
                balanced_accuracy=_balanced_accuracy(rows),
                mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                mean_rounds=sum(rounds) / len(rounds) if rounds else 0.0,
            )

            if (
                baseline_records is not None
                and variant != baseline_name
                and not variant.startswith("baseline")
            ):
                b_correct, v_correct = _paired_records(baseline_records, rows)
                if b_correct:
                    delta, lo, hi, p = _bootstrap_delta_ci(
                        b_correct,
                        v_correct,
                        n_boot=self._bootstrap_n,
                        alpha=self._alpha,
                        seed=self._seed,
                    )
                    stats.delta_acc = delta
                    stats.delta_ci_lower = lo
                    stats.delta_ci_upper = hi
                    stats.p_value = p
                    stats.contribution_label = _label_contribution(delta, lo, hi)

            out.append(stats)

        # Sort: baseline first, then by |Δacc| descending
        out.sort(
            key=lambda s: (
                0 if s.variant.startswith("baseline") else 1,
                -abs(s.delta_acc) if s.delta_acc is not None else 0.0,
            )
        )
        self._stats = out
        return out

    def to_dict(self) -> Dict[str, Any]:
        if not self._stats:
            self.compute()
        return {
            "alpha": self._alpha,
            "bootstrap_n": self._bootstrap_n,
            "variants": [s.to_dict() for s in self._stats],
        }

    def to_markdown(self) -> str:
        """Render a paper-ready Markdown table."""
        if not self._stats:
            self.compute()
        if not self._stats:
            return "_No ablation results to display._"

        lines: List[str] = []
        ab_type = self._stats[0].ablation_type or "ablation"
        lines.append(f"## Ablation report — {ab_type}")
        lines.append("")
        lines.append(
            "| Variant | N | Acc | Bal-Acc | Δacc | 95% CI | p | Latency (ms) | Rounds | Contribution |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|"
        )
        for s in self._stats:
            delta = "—" if s.delta_acc is None else f"{s.delta_acc:+.3f}"
            ci = (
                "—"
                if s.delta_ci_lower is None
                else f"[{s.delta_ci_lower:+.3f}, {s.delta_ci_upper:+.3f}]"
            )
            p = "—" if s.p_value is None else f"{s.p_value:.3f}"
            label = s.contribution_label or "—"
            lines.append(
                f"| `{s.variant}` | {s.n} | {s.accuracy:.3f} | "
                f"{s.balanced_accuracy:.3f} | {delta} | {ci} | {p} | "
                f"{s.mean_latency_ms:.0f} | {s.mean_rounds:.1f} | {label} |"
            )
        lines.append("")
        lines.append(
            f"_Bootstrap n={self._bootstrap_n}, α={self._alpha}, "
            "paired by case_id. Δacc > 0 ⇒ removing the component **hurt** accuracy "
            "(i.e. the component is helpful)._"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DermArbiter ablation analyzer (Tool LOO / Agent LOO / Round)."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to ablation JSONL produced by experiments.ablation.AblationRunner.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for Markdown report. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path for the JSON dict report.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    analyzer = AblationAnalyzer.from_jsonl(
        args.results,
        bootstrap_n=args.bootstrap_n,
        alpha=args.alpha,
        seed=args.seed,
    )
    analyzer.compute()
    md = analyzer.to_markdown()

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        logger.info("Markdown report written to %s", args.output)
    else:
        print(md)

    if args.json:
        Path(args.json).write_text(
            json.dumps(analyzer.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info("JSON report written to %s", args.json)


if __name__ == "__main__":
    main()
