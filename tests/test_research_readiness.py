"""Regression tests for distinguishing development fixtures from research runs."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from dermarbiter.experiments.baseline import BaselineRunner
from dermarbiter.experiments.full_panel import FullPanelRunner
from dermarbiter.experiments.ablation import AblationRunner
from dermarbiter.experiments.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "text_preflight", ROOT / "scripts" / "validate_text_benchmark.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize("runner_type", [BaselineRunner, FullPanelRunner, AblationRunner])
def test_live_mode_cannot_silently_emit_mock_results(runner_type, tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text("existing result\n")
    runner = runner_type(
        config_path="configs/", data_path=str(ROOT / "data/sample_cases.jsonl"),
        output_path=str(output), mock=False,
    )
    with pytest.raises(NotImplementedError):
        runner.run()
    assert output.read_text() == "existing result\n"


@pytest.mark.parametrize(
    "runner_type", [BaselineRunner, FullPanelRunner, AblationRunner, ExperimentRunner]
)
def test_mock_results_are_explicitly_ineligible(runner_type, tmp_path):
    runner = runner_type(
        config_path="configs/", data_path=str(ROOT / "data/sample_cases.jsonl"),
        output_path=str(tmp_path / "results.jsonl"), mock=True, max_cases=1,
    )
    results = runner.run()
    assert results
    assert all(r["execution_mode"] == "mock" for r in results)
    assert all(r["publication_eligible"] is False for r in results)


def valid_case():
    # Schema fixture only. These declarations do not constitute a real reviewed case.
    return {
        "case_id": "fixture-a", "source_id": "article-a", "source": "test_fixture",
        "source_url": "https://example.org/fixture", "license": "fixture only",
        "split": "dev", "modality": "text", "text_origin": "native_clinical_text",
        "clinical_history": "Patient with an itchy plaque for three weeks.",
        "query": "Give a differential diagnosis.", "patient_context": {"age": 30},
        "ground_truth": {"diagnosis_label": "fixture-label"},
        "annotation_status": "frozen", "reviewer_id": "fixture-reviewer",
    }


def test_model_payload_excludes_answers_and_identifiers():
    case = valid_case()
    payload = preflight.model_input(case)
    assert set(payload) == {"clinical_history", "query", "patient_context"}
    assert "fixture-label" not in str(payload)
    assert "article-a" not in str(payload)
    assert preflight.validate_cases([case])["structural_checks_passed"]


@pytest.mark.parametrize("field,value", [
    ("image_path", "lesion.jpg"), ("text_origin", "image_caption"),
    ("annotation_status", "pending"), ("reviewer_id", "auto"),
    ("source", "synthetic"), ("clinical_history", ""),
    ("ground_truth", {}), ("patient_context", {"diagnosis": "answer"}),
    ("split", []), ("text_origin", []),
])
def test_rejects_invalid_primary_track_cases(field, value):
    case = valid_case()
    case[field] = value
    assert not preflight.validate_cases([case])["structural_checks_passed"]


@pytest.mark.parametrize("same_source", [True, False])
def test_rejects_source_or_text_overlap_between_splits(same_source):
    first = valid_case()
    second = copy.deepcopy(first)
    second.update(case_id="fixture-b", split="test")
    if same_source:
        second["clinical_history"] = "Another patient description."
    else:
        second["source_id"] = "article-b"
        second["clinical_history"] = "  PATIENT with an itchy plaque for three weeks.  "
    assert not preflight.validate_cases([first, second])["structural_checks_passed"]


def test_rejects_duplicate_ids_and_empty_dataset():
    assert not preflight.validate_cases([])["structural_checks_passed"]
    assert not preflight.validate_cases([valid_case(), valid_case()])["structural_checks_passed"]


def test_invalid_json_is_not_silently_dropped(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"case_id":\n')
    with pytest.raises(ValueError, match="line 1"):
        preflight.audit_file(path)
