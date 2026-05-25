"""Tests for dermarbiter.evaluation.ablation.AblationAnalyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dermarbiter.evaluation.ablation import (
    AblationAnalyzer,
    VariantStats,
    _accuracy,
    _balanced_accuracy,
    _bootstrap_delta_ci,
    _label_contribution,
    _paired_records,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_records(
    variant: str,
    n: int,
    n_correct: int,
    *,
    ablation_type: str = "tool",
    case_offset: int = 0,
    label: str = "melanoma",
    wrong_label: str = "nevus",
    latency_ms: float = 50.0,
    rounds: float = 1.0,
):
    rows = []
    for i in range(n):
        rows.append({
            "variant": variant,
            "ablation_type": ablation_type,
            "case_id": f"case_{i + case_offset:04d}",
            "predicted": label if i < n_correct else wrong_label,
            "ground_truth": label,
            "latency_ms": latency_ms,
            "num_debate_rounds": rounds,
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_accuracy_basic(self):
        rs = _make_records("v", 10, n_correct=7)
        assert _accuracy(rs) == 0.7

    def test_accuracy_empty(self):
        assert _accuracy([]) == 0.0

    def test_accuracy_case_insensitive(self):
        rs = [{"predicted": "Melanoma", "ground_truth": "melanoma"}]
        assert _accuracy(rs) == 1.0

    def test_accuracy_empty_prediction_counts_as_wrong(self):
        rs = [{"predicted": "", "ground_truth": "melanoma"}]
        assert _accuracy(rs) == 0.0

    def test_balanced_accuracy_two_classes(self):
        rs = (
            _make_records("v", 4, n_correct=2, label="a", wrong_label="b")
            + _make_records("v", 4, n_correct=4, label="b", wrong_label="a", case_offset=10)
        )
        # class a recall = 0.5, class b recall = 1.0 → balanced = 0.75
        assert abs(_balanced_accuracy(rs) - 0.75) < 1e-9

    def test_label_contribution(self):
        assert _label_contribution(0.05, 0.01, 0.10) == "helpful"
        assert _label_contribution(-0.05, -0.10, -0.01) == "harmful"
        assert _label_contribution(0.01, -0.05, 0.05) == "neutral"

    def test_paired_records_aligns_by_case_id(self):
        baseline = [
            {"case_id": "a", "predicted": "x", "ground_truth": "x"},
            {"case_id": "b", "predicted": "x", "ground_truth": "y"},
        ]
        variant = [
            {"case_id": "b", "predicted": "y", "ground_truth": "y"},
            {"case_id": "a", "predicted": "z", "ground_truth": "x"},
        ]
        b, v = _paired_records(baseline, variant)
        # Sorted by case_id: a, b
        assert b == [1, 0]
        assert v == [0, 1]

    def test_bootstrap_delta_ci_signs(self):
        # Baseline always correct, variant always wrong → Δ ≈ +1
        delta, lo, hi, p = _bootstrap_delta_ci(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            n_boot=200,
            seed=0,
        )
        assert delta == 1.0
        assert lo == 1.0 and hi == 1.0
        assert 0.0 <= p <= 1.0

    def test_bootstrap_delta_ci_no_difference(self):
        # Identical performance → Δ ≈ 0, large p
        delta, lo, hi, p = _bootstrap_delta_ci(
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            n_boot=200,
            seed=0,
        )
        assert delta == 0.0
        assert lo == 0.0 and hi == 0.0


# ---------------------------------------------------------------------------
# AblationAnalyzer end-to-end
# ---------------------------------------------------------------------------

class TestAblationAnalyzer:
    def test_empty_records(self):
        analyzer = AblationAnalyzer([])
        stats = analyzer.compute()
        assert stats == []

    def test_baseline_present_and_delta_computed(self):
        baseline = _make_records("baseline_all_tools", n=20, n_correct=16)
        loo = _make_records("no_panderm", n=20, n_correct=10)
        analyzer = AblationAnalyzer(baseline + loo, bootstrap_n=200, seed=42)
        stats = analyzer.compute()

        by_name = {s.variant: s for s in stats}
        assert "baseline_all_tools" in by_name
        assert "no_panderm" in by_name

        b = by_name["baseline_all_tools"]
        v = by_name["no_panderm"]
        assert b.accuracy == 0.80
        assert v.accuracy == 0.50
        # Δacc = baseline − variant ≈ +0.30
        assert v.delta_acc is not None
        assert abs(v.delta_acc - 0.30) < 1e-9
        # CI should include the point estimate
        assert v.delta_ci_lower <= v.delta_acc <= v.delta_ci_upper
        # Baseline itself gets no delta
        assert b.delta_acc is None

    def test_baseline_sorts_first(self):
        rows = (
            _make_records("no_panderm", 5, 2)
            + _make_records("baseline_all_tools", 5, 5, case_offset=100)
            + _make_records("no_make", 5, 3, case_offset=200)
        )
        analyzer = AblationAnalyzer(rows, bootstrap_n=100, seed=0)
        stats = analyzer.compute()
        assert stats[0].variant.startswith("baseline")

    def test_contribution_label_helpful(self):
        # Removing a tool drops accuracy a lot — should be flagged "helpful"
        baseline = _make_records("baseline_all_tools", 30, 28)
        loo = _make_records("no_critical", 30, 5)
        analyzer = AblationAnalyzer(baseline + loo, bootstrap_n=500, seed=7)
        stats = {s.variant: s for s in analyzer.compute()}
        assert stats["no_critical"].contribution_label == "helpful"
        assert stats["no_critical"].delta_acc > 0

    def test_to_markdown_contains_variants(self):
        rows = (
            _make_records("baseline_all_tools", 10, 8)
            + _make_records("no_x", 10, 5)
        )
        analyzer = AblationAnalyzer(rows, bootstrap_n=100, seed=0)
        md = analyzer.to_markdown()
        assert "baseline_all_tools" in md
        assert "no_x" in md
        assert "Δacc" in md

    def test_to_dict_round_trip(self):
        rows = (
            _make_records("baseline_all_tools", 10, 8)
            + _make_records("no_x", 10, 5)
        )
        analyzer = AblationAnalyzer(rows, bootstrap_n=100, seed=0)
        d = analyzer.to_dict()
        assert "variants" in d
        assert "alpha" in d
        assert isinstance(d["variants"], list)
        # JSON-serializable
        json.dumps(d)

    def test_from_jsonl_round_trip(self, tmp_path: Path):
        rows = (
            _make_records("baseline_all_tools", 8, 7)
            + _make_records("no_y", 8, 4)
        )
        p = tmp_path / "ab.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        analyzer = AblationAnalyzer.from_jsonl(p, bootstrap_n=100, seed=0)
        stats = analyzer.compute()
        assert len(stats) == 2

    def test_handles_missing_case_id_gracefully(self):
        rows = [
            {"variant": "baseline_all_tools", "ablation_type": "tool",
             "predicted": "a", "ground_truth": "a"},
            {"variant": "no_x", "ablation_type": "tool",
             "predicted": "b", "ground_truth": "a"},
        ]
        # No case_id → paired bootstrap skipped, but stats still computed
        analyzer = AblationAnalyzer(rows, bootstrap_n=50, seed=0)
        stats = analyzer.compute()
        names = {s.variant for s in stats}
        assert names == {"baseline_all_tools", "no_x"}
