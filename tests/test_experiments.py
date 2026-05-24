"""Tests for the DermArbiter experiments pipeline.

Covers:
    • BenchmarkRunner — JSONL loading, mock pipeline execution, output format
    • ResultsAnalyzer — accuracy, F1, calibration, efficiency metrics
    • AblationRunner  — config parsing and variant generation
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dermarbiter.experiments.runner import BenchmarkRunner, _load_cases
from dermarbiter.experiments.analyze import ResultsAnalyzer, _compute_f1, _compute_ece, _compute_brier
from dermarbiter.experiments.ablation import (
    AblationConfig,
    AblationRunner,
    AGENT_ABLATION,
    TOOL_ABLATION,
    ROUND_ABLATION,
    VALID_ABLATION_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Path to sample data shipped with the project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_DATA = _PROJECT_ROOT / "data" / "sample_cases.jsonl"
_CONFIG_DIR = _PROJECT_ROOT / "configs" / "default.yaml"


def _make_temp_jsonl(records: List[Dict[str, Any]]) -> str:
    """Write records to a temp JSONL file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def _sample_results() -> List[Dict[str, Any]]:
    """Create synthetic benchmark results for analyzer tests."""
    return [
        {
            "case_id": "T-001",
            "predicted": "melanoma",
            "ground_truth": "melanoma",
            "final_diagnosis": ["melanoma", "dysplastic_nevus", "basal_cell_carcinoma"],
            "consensus_score": 0.90,
            "early_exit": False,
            "num_debate_rounds": 3,
            "total_tokens": 12000,
            "latency_ms": 450.0,
        },
        {
            "case_id": "T-002",
            "predicted": "melanoma",
            "ground_truth": "dysplastic_nevus",
            "final_diagnosis": ["melanoma", "dysplastic_nevus", "seborrheic_keratosis"],
            "consensus_score": 0.80,
            "early_exit": False,
            "num_debate_rounds": 3,
            "total_tokens": 11500,
            "latency_ms": 420.0,
        },
        {
            "case_id": "T-003",
            "predicted": "melanoma",
            "ground_truth": "melanoma",
            "final_diagnosis": ["melanoma", "basal_cell_carcinoma"],
            "consensus_score": 0.95,
            "early_exit": True,
            "num_debate_rounds": 0,
            "total_tokens": 8000,
            "latency_ms": 300.0,
        },
        {
            "case_id": "T-004",
            "predicted": "basal_cell_carcinoma",
            "ground_truth": "basal_cell_carcinoma",
            "final_diagnosis": ["basal_cell_carcinoma", "melanoma"],
            "consensus_score": 0.85,
            "early_exit": False,
            "num_debate_rounds": 2,
            "total_tokens": 10000,
            "latency_ms": 380.0,
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# BenchmarkRunner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner with mock agents."""

    def test_load_cases_from_sample_file(self):
        """_load_cases should parse the shipped sample_cases.jsonl correctly."""
        cases = _load_cases(str(_SAMPLE_DATA))
        assert len(cases) == 5
        assert all("case_id" in c for c in cases)
        assert all("ground_truth_label" in c for c in cases)

    def test_load_cases_max_cases_limit(self):
        """_load_cases with max_cases should truncate the result list."""
        cases = _load_cases(str(_SAMPLE_DATA), max_cases=2)
        assert len(cases) == 2

    def test_run_mock_pipeline_produces_results(self, tmp_path):
        """Full mock pipeline run should produce one result per input case."""
        output = str(tmp_path / "results.jsonl")
        runner = BenchmarkRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            output_path=output,
            mock=True,
            max_cases=2,
        )
        results = runner.run()

        assert len(results) == 2
        assert os.path.exists(output)

        # Verify JSONL output
        with open(output, "r") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        assert len(lines) == 2

        for r in results:
            assert "case_id" in r
            assert "predicted" in r
            assert "ground_truth" in r
            assert "consensus_score" in r
            assert "latency_ms" in r
            assert "total_tokens" in r

    def test_run_mock_pipeline_result_fields(self, tmp_path):
        """Each result should contain all expected telemetry fields."""
        output = str(tmp_path / "results.jsonl")
        runner = BenchmarkRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            output_path=output,
            mock=True,
            max_cases=1,
        )
        results = runner.run()
        r = results[0]

        expected_keys = {
            "case_id", "predicted", "ground_truth", "final_diagnosis",
            "consensus_score", "early_exit", "num_debate_rounds",
            "total_tokens", "latency_ms",
        }
        assert expected_keys.issubset(set(r.keys()))
        assert isinstance(r["final_diagnosis"], list)
        assert isinstance(r["consensus_score"], float)
        assert r["latency_ms"] > 0

    def test_run_mock_pipeline_predicted_not_empty(self, tmp_path):
        """Mock agents should always produce a non-empty predicted diagnosis."""
        output = str(tmp_path / "results.jsonl")
        runner = BenchmarkRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            output_path=output,
            mock=True,
            max_cases=3,
        )
        results = runner.run()
        for r in results:
            assert r["predicted"], f"Empty prediction for case {r['case_id']}"

    def test_load_cases_skips_blank_lines(self, tmp_path):
        """_load_cases should skip blank lines gracefully."""
        data_file = tmp_path / "test.jsonl"
        data_file.write_text(
            '{"case_id":"A","query":"q","image_path":null,"patient_context":{},"ground_truth_label":"x"}\n'
            "\n"
            '{"case_id":"B","query":"q2","image_path":null,"patient_context":{},"ground_truth_label":"y"}\n'
        )
        cases = _load_cases(str(data_file))
        assert len(cases) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ResultsAnalyzer Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResultsAnalyzer:
    """Tests for ResultsAnalyzer with synthetic data."""

    def test_accuracy_computation(self):
        """Accuracy should equal num_correct / total."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        # 3 correct out of 4: T-001 ✓, T-002 ✗, T-003 ✓, T-004 ✓
        assert analyzer.accuracy() == pytest.approx(3 / 4)

    def test_top3_accuracy(self):
        """Top-3 accuracy: ground_truth in final_diagnosis[:3]."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        # T-001: melanoma in [melanoma, dysplastic_nevus, bcc] ✓
        # T-002: dysplastic_nevus in [melanoma, dysplastic_nevus, seb_k] ✓
        # T-003: melanoma in [melanoma, bcc] ✓
        # T-004: bcc in [bcc, melanoma] ✓
        assert analyzer.top3_accuracy() == pytest.approx(1.0)

    def test_confusion_matrix_structure(self):
        """Confusion matrix should have the right shape and counts."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        cm = analyzer.confusion_matrix()
        # melanoma is true for T-001, T-003 → predicted melanoma for both
        assert cm["melanoma"]["melanoma"] == 2
        # dysplastic_nevus true for T-002 → predicted melanoma
        assert cm["dysplastic_nevus"]["melanoma"] == 1

    def test_per_class_f1(self):
        """Per-class F1 for melanoma should be computable."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        f1s = analyzer.per_class_f1()
        assert "melanoma" in f1s
        # melanoma: TP=2, FP=1 (T-002), FN=0 → P=2/3, R=1.0, F1=0.8
        assert f1s["melanoma"] == pytest.approx(0.8, abs=0.01)

    def test_macro_and_weighted_f1(self):
        """Macro and weighted F1 should be between 0 and 1."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        assert 0.0 <= analyzer.macro_f1() <= 1.0
        assert 0.0 <= analyzer.weighted_f1() <= 1.0

    def test_calibration_ece(self):
        """ECE should be between 0 and 1."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        ece = analyzer.ece()
        assert 0.0 <= ece <= 1.0

    def test_brier_score(self):
        """Brier score should be between 0 and 1."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        brier = analyzer.brier_score()
        assert 0.0 <= brier <= 1.0

    def test_efficiency_metrics(self):
        """Efficiency metrics should be non-negative."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        assert analyzer.early_exit_rate() == pytest.approx(1 / 4)
        assert analyzer.avg_debate_rounds() == pytest.approx((3 + 3 + 0 + 2) / 4)
        assert analyzer.avg_tokens() > 0
        assert analyzer.avg_latency_ms() > 0

    def test_to_dict_has_all_keys(self):
        """to_dict() should contain all expected metric keys."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        d = analyzer.to_dict()
        expected_keys = {
            "n_cases", "accuracy", "top3_accuracy", "per_class_f1",
            "macro_f1", "weighted_f1", "ece", "brier_score",
            "confusion_matrix", "early_exit_rate", "avg_debate_rounds",
            "avg_tokens", "avg_latency_ms",
        }
        assert expected_keys == set(d.keys())

    def test_empty_records(self):
        """Analyzer should handle empty records gracefully."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records([])
        assert analyzer.accuracy() == 0.0
        assert analyzer.top3_accuracy() == 0.0
        assert analyzer.macro_f1() == 0.0

    def test_load_from_file(self, tmp_path):
        """Analyzer should load from a JSONL file correctly."""
        path = _make_temp_jsonl(_sample_results())
        try:
            analyzer = ResultsAnalyzer(results_path=path)
            assert len(analyzer.records) == 4
            assert analyzer.accuracy() == pytest.approx(3 / 4)
        finally:
            os.unlink(path)

    def test_print_report_runs(self, capsys):
        """print_report() should complete without errors."""
        analyzer = ResultsAnalyzer()
        analyzer.load_records(_sample_results())
        analyzer.print_report()
        captured = capsys.readouterr()
        assert "DermArbiter Benchmark Report" in captured.out
        assert "Accuracy" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# Metric Helper Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricHelpers:
    """Tests for low-level metric computation functions."""

    def test_f1_perfect(self):
        """Perfect classification should yield F1 = 1.0."""
        assert _compute_f1(tp=10, fp=0, fn=0) == pytest.approx(1.0)

    def test_f1_zero(self):
        """Zero TP with non-zero FP/FN should yield F1 = 0.0."""
        assert _compute_f1(tp=0, fp=5, fn=3) == pytest.approx(0.0)

    def test_ece_perfect_calibration(self):
        """When confidence matches accuracy exactly, ECE should be ~0."""
        confidences = [1.0, 1.0, 0.0, 0.0]
        accuracies = [True, True, False, False]
        assert _compute_ece(confidences, accuracies) == pytest.approx(0.0, abs=0.01)

    def test_brier_perfect(self):
        """Perfect confident predictions should give Brier = 0."""
        confidences = [1.0, 1.0, 1.0]
        accuracies = [True, True, True]
        assert _compute_brier(confidences, accuracies) == pytest.approx(0.0)

    def test_brier_worst(self):
        """Maximally wrong confident predictions should give Brier = 1."""
        confidences = [1.0, 1.0]
        accuracies = [False, False]
        assert _compute_brier(confidences, accuracies) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# AblationRunner Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAblationRunner:
    """Tests for AblationRunner config parsing and variant generation."""

    def test_ablation_config_valid_types(self):
        """AblationConfig should accept all valid ablation types."""
        for atype in VALID_ABLATION_TYPES:
            cfg = AblationConfig(ablation_type=atype, variant_name=f"test_{atype}")
            assert cfg.ablation_type == atype

    def test_ablation_config_invalid_type_raises(self):
        """AblationConfig should reject invalid ablation types."""
        with pytest.raises(ValueError, match="Invalid ablation_type"):
            AblationConfig(ablation_type="invalid", variant_name="bad")

    def test_agent_ablation_variants(self):
        """Agent ablation should produce baseline + 3 remove-one variants."""
        runner = AblationRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            ablation_type=AGENT_ABLATION,
            mock=True,
        )
        variants = runner._generate_variants()
        # baseline + no_specialist + no_generalist + no_skeptic
        assert len(variants) == 4
        names = {v.variant_name for v in variants}
        assert "baseline_all_agents" in names
        assert "no_specialist" in names
        assert "no_generalist" in names
        assert "no_skeptic" in names

    def test_round_ablation_variants(self):
        """Round ablation should produce one variant per round value."""
        runner = AblationRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            ablation_type=ROUND_ABLATION,
            mock=True,
            round_values=[1, 3, 5],
        )
        variants = runner._generate_variants()
        assert len(variants) == 3
        assert variants[0].max_rounds == 1
        assert variants[1].max_rounds == 3
        assert variants[2].max_rounds == 5

    def test_tool_ablation_variants(self):
        """Tool ablation should produce baseline + N variants (one per tool)."""
        runner = AblationRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            ablation_type=TOOL_ABLATION,
            mock=True,
        )
        variants = runner._generate_variants()
        # baseline + 9 tools = 10 variants
        assert len(variants) == 10
        assert variants[0].variant_name == "baseline_all_tools"

    def test_ablation_config_to_dict(self):
        """AblationConfig.to_dict() should serialize all fields."""
        cfg = AblationConfig(
            ablation_type=AGENT_ABLATION,
            variant_name="no_skeptic",
            removed_agents=["skeptic"],
        )
        d = cfg.to_dict()
        assert d["ablation_type"] == "agent"
        assert d["variant_name"] == "no_skeptic"
        assert d["removed_agents"] == ["skeptic"]

    def test_ablation_runner_invalid_type_raises(self):
        """AblationRunner should reject invalid ablation types at init."""
        with pytest.raises(ValueError, match="Invalid ablation_type"):
            AblationRunner(
                config_path=str(_CONFIG_DIR),
                data_path=str(_SAMPLE_DATA),
                ablation_type="invalid",
            )

    def test_ablation_run_mock_agent(self, tmp_path):
        """Full agent ablation should produce results for all variants × cases."""
        output = str(tmp_path / "ablation_results.jsonl")
        runner = AblationRunner(
            config_path=str(_CONFIG_DIR),
            data_path=str(_SAMPLE_DATA),
            ablation_type=AGENT_ABLATION,
            output_path=output,
            mock=True,
            max_cases=1,
        )
        results = runner.run()
        # 4 variants × 1 case = 4 results
        assert len(results) == 4
        assert os.path.exists(output)

        # Verify all variants are represented
        variant_names = {r["variant"] for r in results}
        assert "baseline_all_agents" in variant_names
