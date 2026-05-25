"""Tests for the dermarbiter.evaluation module.

Covers:
    - MetricsCalculator: accuracy, balanced accuracy, top-k, F1, ECE, Brier, etc.
    - FairnessAnalyzer: per-group metrics, equalized odds, demographic parity
    - BenchmarkRunner: setup, dataset loading, case execution
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dermarbiter.evaluation.metrics import MetricsCalculator
from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer, GroupMetrics
from dermarbiter.evaluation.benchmark_runner import BenchmarkRunner, DatasetLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RECORDS: List[Dict[str, Any]] = [
    {
        "case_id": "c1", "predicted": "melanoma", "ground_truth": "melanoma",
        "final_diagnosis": ["melanoma", "bcc", "nv"],
        "consensus_score": 0.85, "early_exit": False, "num_debate_rounds": 2,
        "total_tokens": 1200, "total_tool_calls": 5, "latency_ms": 340.0,
        "fitzpatrick_type": "II",
    },
    {
        "case_id": "c2", "predicted": "bcc", "ground_truth": "bcc",
        "final_diagnosis": ["bcc", "melanoma"],
        "consensus_score": 0.78, "early_exit": True, "num_debate_rounds": 0,
        "total_tokens": 800, "total_tool_calls": 3, "latency_ms": 210.0,
        "fitzpatrick_type": "III",
    },
    {
        "case_id": "c3", "predicted": "melanoma", "ground_truth": "nv",
        "final_diagnosis": ["melanoma", "nv", "bkl"],
        "consensus_score": 0.62, "early_exit": False, "num_debate_rounds": 3,
        "total_tokens": 2100, "total_tool_calls": 7, "latency_ms": 580.0,
        "fitzpatrick_type": "V",
    },
    {
        "case_id": "c4", "predicted": "nv", "ground_truth": "nv",
        "final_diagnosis": ["nv", "bkl"],
        "consensus_score": 0.91, "early_exit": True, "num_debate_rounds": 0,
        "total_tokens": 600, "total_tool_calls": 2, "latency_ms": 150.0,
        "fitzpatrick_type": "II",
    },
    {
        "case_id": "c5", "predicted": "bcc", "ground_truth": "melanoma",
        "final_diagnosis": ["bcc", "melanoma"],
        "consensus_score": 0.55, "early_exit": False, "num_debate_rounds": 3,
        "total_tokens": 2500, "total_tool_calls": 8, "latency_ms": 620.0,
        "fitzpatrick_type": "V",
    },
    {
        "case_id": "c6", "predicted": "bkl", "ground_truth": "bkl",
        "final_diagnosis": ["bkl", "nv"],
        "consensus_score": 0.88, "early_exit": True, "num_debate_rounds": 0,
        "total_tokens": 700, "total_tool_calls": 3, "latency_ms": 180.0,
        "fitzpatrick_type": "IV",
    },
]


@pytest.fixture
def sample_records():
    return [dict(r) for r in SAMPLE_RECORDS]


@pytest.fixture
def sample_jsonl(sample_records, tmp_path):
    path = tmp_path / "test_results.jsonl"
    with open(path, "w") as fh:
        for r in sample_records:
            fh.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def calculator(sample_records):
    return MetricsCalculator(records=sample_records)


@pytest.fixture
def fairness_analyzer(sample_records):
    return FairnessAnalyzer(records=sample_records, group_key="fitzpatrick_type")


# ===========================================================================
# MetricsCalculator Tests
# ===========================================================================

class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_from_jsonl(self, sample_jsonl):
        calc = MetricsCalculator.from_jsonl(sample_jsonl)
        assert calc.n_cases == 6

    def test_accuracy(self, calculator):
        # 4 correct out of 6: c1, c2, c4, c6
        assert calculator.accuracy() == pytest.approx(4 / 6)

    def test_balanced_accuracy(self, calculator):
        ba = calculator.balanced_accuracy()
        # Recall per class: melanoma=1/2, bcc=1/1, nv=1/2, bkl=1/1
        # balanced = mean(0.5, 1.0, 0.5, 1.0) = 0.75
        assert ba == pytest.approx(0.75)

    def test_topk_accuracy_3(self, calculator):
        # All ground truths appear in top-3 final_diagnosis
        assert calculator.topk_accuracy(k=3) == pytest.approx(1.0)

    def test_topk_accuracy_1(self, calculator):
        # Same as regular accuracy when k=1
        assert calculator.topk_accuracy(k=1) == pytest.approx(4 / 6)

    def test_per_class_f1(self, calculator):
        f1s = calculator.per_class_f1()
        assert "melanoma" in f1s
        assert "bcc" in f1s
        assert "nv" in f1s
        assert "bkl" in f1s
        # All F1 values should be in [0, 1]
        for v in f1s.values():
            assert 0.0 <= v <= 1.0

    def test_macro_f1(self, calculator):
        mf1 = calculator.macro_f1()
        assert 0.0 <= mf1 <= 1.0
        # Should be mean of per-class F1
        f1s = calculator.per_class_f1()
        expected = sum(f1s.values()) / len(f1s)
        assert mf1 == pytest.approx(expected)

    def test_weighted_f1(self, calculator):
        wf1 = calculator.weighted_f1()
        assert 0.0 <= wf1 <= 1.0

    def test_sensitivity(self, calculator):
        sens = calculator.sensitivity()
        # melanoma: 1 TP / 2 positives = 0.5
        assert sens["melanoma"] == pytest.approx(0.5)
        # bcc: 1 TP / 1 positive = 1.0
        assert sens["bcc"] == pytest.approx(1.0)

    def test_specificity(self, calculator):
        spec = calculator.specificity()
        for v in spec.values():
            assert 0.0 <= v <= 1.0

    def test_ece(self, calculator):
        ece = calculator.ece()
        assert 0.0 <= ece <= 1.0

    def test_brier_score(self, calculator):
        bs = calculator.brier_score()
        assert 0.0 <= bs <= 1.0

    def test_confusion_matrix(self, calculator):
        cm = calculator.confusion_matrix()
        assert isinstance(cm, dict)
        # melanoma ground truth: 1 predicted as melanoma, 1 as bcc
        assert cm["melanoma"]["melanoma"] == 1
        assert cm["melanoma"]["bcc"] == 1

    def test_early_exit_rate(self, calculator):
        # 3 early exits out of 6
        assert calculator.early_exit_rate() == pytest.approx(0.5)

    def test_avg_debate_rounds(self, calculator):
        # (2 + 0 + 3 + 0 + 3 + 0) / 6 = 8/6
        assert calculator.avg_debate_rounds() == pytest.approx(8 / 6)

    def test_avg_tokens(self, calculator):
        expected = (1200 + 800 + 2100 + 600 + 2500 + 700) / 6
        assert calculator.avg_tokens() == pytest.approx(expected)

    def test_avg_latency_ms(self, calculator):
        expected = (340 + 210 + 580 + 150 + 620 + 180) / 6
        assert calculator.avg_latency_ms() == pytest.approx(expected)

    def test_avg_tool_calls(self, calculator):
        expected = (5 + 3 + 7 + 2 + 8 + 3) / 6
        assert calculator.avg_tool_calls() == pytest.approx(expected)

    def test_latency_p50(self, calculator):
        # sorted latencies: 150, 180, 210, 340, 580, 620 → median = (210+340)/2 = 275
        assert calculator.latency_p50() == pytest.approx(275.0)

    def test_latency_p95(self, calculator):
        # numpy default linear interp; should sit close to the 95th percentile
        p95 = calculator.latency_p95()
        assert 600.0 <= p95 <= 620.0

    def test_throughput_cases_per_min(self, calculator):
        avg = calculator.avg_latency_ms()
        assert calculator.throughput_cases_per_min() == pytest.approx(60_000.0 / avg)

    def test_throughput_empty(self):
        calc = MetricsCalculator(records=[])
        assert calc.throughput_cases_per_min() == 0.0

    def test_avg_cost_usd_scales_with_token_price(self, calculator):
        toks = calculator.avg_tokens()
        # default price = 0.00015 USD / 1k
        expected = (toks / 1000.0) * 0.00015
        assert calculator.avg_cost_usd() == pytest.approx(expected)
        # Override price → linear scaling
        assert calculator.avg_cost_usd(cost_per_1k_tokens=0.0003) == pytest.approx(2 * expected)

    def test_delta_accuracy(self, calculator):
        da = calculator.delta_accuracy(group_key="fitzpatrick_type")
        assert "per_group" in da
        assert "max_delta" in da
        # Group V has 0% accuracy, others 100%
        assert da["max_delta"] == pytest.approx(1.0)

    def test_compute_all(self, calculator):
        result = calculator.compute_all()
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "ece" in result
        assert "brier_score" in result
        assert "early_exit_rate" in result

    def test_compute_all_with_fairness(self, calculator):
        result = calculator.compute_all(include_fairness=True)
        assert "delta_accuracy" in result

    def test_empty_calculator(self):
        calc = MetricsCalculator(records=[])
        assert calc.accuracy() == 0.0
        assert calc.balanced_accuracy() == 0.0
        assert calc.topk_accuracy() == 0.0
        assert calc.macro_f1() == 0.0
        assert calc.ece() == 0.0
        assert calc.brier_score() == 0.0
        assert calc.early_exit_rate() == 0.0

    def test_class_names_inferred(self, calculator):
        names = calculator.class_names
        assert set(names) == {"melanoma", "bcc", "nv", "bkl"}

    def test_class_names_explicit(self, sample_records):
        calc = MetricsCalculator(records=sample_records, class_names=["melanoma", "bcc"])
        assert calc.class_names == ["melanoma", "bcc"]


# ===========================================================================
# FairnessAnalyzer Tests
# ===========================================================================

class TestFairnessAnalyzer:
    """Tests for FairnessAnalyzer."""

    def test_from_jsonl(self, sample_jsonl):
        analyzer = FairnessAnalyzer.from_jsonl(sample_jsonl)
        assert len(analyzer._records) == 6

    def test_group_names(self, fairness_analyzer):
        names = fairness_analyzer.group_names
        # Should follow Fitzpatrick order for known types
        assert names == ["II", "III", "IV", "V"]

    def test_per_group_accuracy(self, fairness_analyzer):
        accs = fairness_analyzer.per_group_accuracy()
        assert accs["II"] == pytest.approx(1.0)   # 2/2 correct
        assert accs["III"] == pytest.approx(1.0)   # 1/1 correct
        assert accs["IV"] == pytest.approx(1.0)    # 1/1 correct
        assert accs["V"] == pytest.approx(0.0)     # 0/2 correct

    def test_per_group_n(self, fairness_analyzer):
        ns = fairness_analyzer.per_group_n()
        assert ns["II"] == 2
        assert ns["V"] == 2

    def test_accuracy_gap(self, fairness_analyzer):
        ag = fairness_analyzer.accuracy_gap()
        assert ag["max_delta"] == pytest.approx(1.0)
        assert ag["min_group"] == "V"

    def test_equalized_odds(self, fairness_analyzer):
        eo = fairness_analyzer.equalized_odds()
        assert "max_tpr_disparity" in eo
        assert "max_fpr_disparity" in eo
        assert "satisfied" in eo
        # With such extreme accuracy gap, should NOT be satisfied
        assert eo["satisfied"] is False

    def test_demographic_parity(self, fairness_analyzer):
        dp = fairness_analyzer.demographic_parity()
        assert "max_disparity" in dp
        assert "satisfied" in dp

    def test_calibration_gap(self, fairness_analyzer):
        cg = fairness_analyzer.calibration_gap()
        assert "per_group_ece" in cg
        assert "max_gap" in cg
        assert cg["max_gap"] >= 0.0

    def test_compute_all(self, fairness_analyzer):
        report = fairness_analyzer.compute_all()
        assert report["n_total"] == 6
        assert report["n_groups"] == 4
        assert "equalized_odds" in report
        assert "demographic_parity" in report
        assert "calibration_gap" in report

    def test_empty_analyzer(self):
        analyzer = FairnessAnalyzer(records=[])
        report = analyzer.compute_all()
        assert report["n_total"] == 0
        assert report["n_groups"] == 0

    def test_single_group(self):
        records = [
            {"predicted": "mel", "ground_truth": "mel", "consensus_score": 0.9, "group": "A"},
        ]
        analyzer = FairnessAnalyzer(records=records, group_key="group")
        eo = analyzer.equalized_odds()
        assert eo["satisfied"] is True  # Only 1 group, trivially satisfied


# ===========================================================================
# GroupMetrics Tests
# ===========================================================================

class TestGroupMetrics:
    """Tests for GroupMetrics helper class."""

    def test_accuracy(self):
        gm = GroupMetrics("test", ["a", "b", "a"], ["a", "a", "a"], [0.9, 0.7, 0.8])
        assert gm.accuracy == pytest.approx(2 / 3)

    def test_sensitivity(self):
        gm = GroupMetrics("test", ["a", "a", "b"], ["a", "b", "b"], [0.9, 0.7, 0.8])
        # For class "a": TP=1, FN=1 → sensitivity=0.5
        assert gm.sensitivity_for_class("a") == pytest.approx(0.5)

    def test_specificity(self):
        gm = GroupMetrics("test", ["a", "a", "b"], ["a", "b", "b"], [0.9, 0.7, 0.8])
        # For class "a": TN=1, FP=0 → specificity=1.0
        assert gm.specificity_for_class("a") == pytest.approx(1.0)

    def test_f1(self):
        gm = GroupMetrics("test", ["a", "a", "b"], ["a", "a", "b"], [0.9, 0.9, 0.8])
        # Perfect predictions → F1=1.0 for both classes
        assert gm.f1_for_class("a") == pytest.approx(1.0)
        assert gm.f1_for_class("b") == pytest.approx(1.0)

    def test_empty_group(self):
        gm = GroupMetrics("empty", [], [], [])
        assert gm.accuracy == 0.0
        assert gm.n == 0


# ===========================================================================
# DatasetLoader Tests
# ===========================================================================

class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_jsonl(self, sample_jsonl):
        cases = DatasetLoader.load_jsonl(sample_jsonl)
        assert len(cases) == 6

    def test_load_jsonl_max_cases(self, sample_jsonl):
        cases = DatasetLoader.load_jsonl(sample_jsonl, max_cases=3)
        assert len(cases) == 3

    def test_load_jsonl_missing_file(self, tmp_path):
        cases = DatasetLoader.load_jsonl(tmp_path / "nonexistent.jsonl")
        assert cases == []

    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w") as fh:
            fh.write("case_id,ground_truth_label,image_path\n")
            fh.write("c1,melanoma,img1.jpg\n")
            fh.write("c2,bcc,img2.jpg\n")
        cases = DatasetLoader.load_csv(csv_path)
        assert len(cases) == 2
        assert cases[0]["case_id"] == "c1"

    def test_load_generic_jsonl(self, sample_jsonl, tmp_path):
        # Copy to expected location
        target = tmp_path / "test.jsonl"
        import shutil
        shutil.copy(sample_jsonl, target)
        cases = DatasetLoader.load_generic(tmp_path, split="test")
        assert len(cases) == 6


# ===========================================================================
# BenchmarkRunner Tests
# ===========================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_init(self, tmp_path):
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        # Create minimal benchmarks.yaml
        bench_yaml = config_dir / "benchmarks.yaml"
        bench_yaml.write_text(
            "benchmarks:\n"
            "  test_bench:\n"
            "    name: TestBench\n"
            "    data_dir: data/test/\n"
            "    split: test\n"
            "    task: classification\n"
        )
        runner = BenchmarkRunner(config_dir=config_dir, mock=True)
        assert "test_bench" in runner.available_benchmarks

    def test_available_benchmarks(self):
        runner = BenchmarkRunner(config_dir="configs/", mock=True)
        benchmarks = runner.available_benchmarks
        assert isinstance(benchmarks, list)
        # Should have benchmarks from configs/benchmarks.yaml
        assert len(benchmarks) > 0

    def test_unknown_benchmark_raises(self):
        runner = BenchmarkRunner(config_dir="configs/", mock=True)
        with pytest.raises(ValueError, match="Unknown benchmark"):
            runner.run_benchmark("nonexistent_benchmark")
