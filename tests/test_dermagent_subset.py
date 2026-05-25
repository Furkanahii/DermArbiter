"""Tests for the DermAgent-comparison scripts.

Covers:
    scripts/build_dermagent_subset.py   — filtering HAM10000 to DermAgent's 642 IDs
    scripts/run_dermagent_subset.py     — mock evaluator wiring

Both are exercised against tiny synthetic data so the tests run in a few
hundred milliseconds with no network and no large fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_dermagent_subset as bds  # noqa: E402
import run_dermagent_subset as rds  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# build_dermagent_subset
# ─────────────────────────────────────────────────────────────────────────────


def _write_ham_metadata(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a minimal HAM10000_metadata.csv with (lesion_id, image_id, dx) tuples."""
    lines = ["lesion_id,image_id,dx,dx_type,age,sex,localization,dataset"]
    for lesion, image, dx in rows:
        lines.append(f"{lesion},{image},{dx},histo,55,male,back,vidir_modern")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subset_csv(path: Path, image_ids: list[str]) -> None:
    """Write a DermAgent-style subset CSV with just the image_id column populated."""
    lines = ["lesion_id,image_id,dx,dx_type,age,sex,localization"]
    for iid in image_ids:
        lines.append(f"HAM_X,{iid},nv,histo,30,male,back")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestLoadSubsetIds:
    def test_reads_image_ids(self, tmp_path: Path):
        csv = tmp_path / "subset.csv"
        _write_subset_csv(csv, ["ISIC_001", "ISIC_002", "ISIC_003"])
        ids = bds.load_subset_ids(csv)
        assert ids == {"ISIC_001", "ISIC_002", "ISIC_003"}

    def test_deduplicates(self, tmp_path: Path):
        csv = tmp_path / "subset.csv"
        _write_subset_csv(csv, ["ISIC_001", "ISIC_001"])
        assert bds.load_subset_ids(csv) == {"ISIC_001"}

    def test_raises_when_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            bds.load_subset_ids(tmp_path / "nope.csv")


class TestBuild:
    def test_filters_to_subset(self, tmp_path: Path):
        _write_ham_metadata(tmp_path / "HAM10000_metadata.csv", [
            ("L1", "ISIC_001", "mel"),
            ("L2", "ISIC_002", "nv"),
            ("L3", "ISIC_003", "bcc"),
            ("L4", "ISIC_004", "df"),
        ])
        _write_subset_csv(tmp_path / "subset.csv", ["ISIC_002", "ISIC_004"])

        stats = bds.build(
            ham_dir=tmp_path,
            subset_csv=tmp_path / "subset.csv",
            output_path=tmp_path / "out.jsonl",
            require_image=False,
        )

        assert stats.requested == 2
        assert stats.written == 2
        assert stats.missing_in_ham10000 == []

        out_lines = (tmp_path / "out.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in out_lines]
        kept_ids = {r["case_id"] for r in records}
        assert kept_ids == {"ISIC_002", "ISIC_004"}
        # Schema check on one record.
        r = next(r for r in records if r["case_id"] == "ISIC_004")
        assert r["ground_truth_label"] == "df"
        assert r["patient_context"]["localization"] == "back"
        assert r["subset_source"] == "DermAgent_HAM10000_benchmark_500"

    def test_reports_subset_ids_missing_from_ham10000(self, tmp_path: Path):
        _write_ham_metadata(tmp_path / "HAM10000_metadata.csv", [
            ("L1", "ISIC_001", "mel"),
        ])
        _write_subset_csv(tmp_path / "subset.csv", ["ISIC_001", "ISIC_999"])

        stats = bds.build(
            ham_dir=tmp_path,
            subset_csv=tmp_path / "subset.csv",
            output_path=tmp_path / "out.jsonl",
            require_image=False,
        )
        assert stats.written == 1
        assert stats.missing_in_ham10000 == ["ISIC_999"]

    def test_require_image_drops_missing_files(self, tmp_path: Path):
        _write_ham_metadata(tmp_path / "HAM10000_metadata.csv", [
            ("L1", "ISIC_001", "mel"),
            ("L2", "ISIC_002", "nv"),
        ])
        _write_subset_csv(tmp_path / "subset.csv", ["ISIC_001", "ISIC_002"])
        # Only one image actually exists on disk.
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "ISIC_001.jpg").write_bytes(b"x")

        stats = bds.build(
            ham_dir=tmp_path,
            subset_csv=tmp_path / "subset.csv",
            output_path=tmp_path / "out.jsonl",
            require_image=True,
        )
        assert stats.written == 1
        assert stats.missing_image_file == ["ISIC_002"]

    def test_raises_when_ham_metadata_missing(self, tmp_path: Path):
        _write_subset_csv(tmp_path / "subset.csv", ["ISIC_001"])
        with pytest.raises(FileNotFoundError, match="HAM10000_metadata.csv"):
            bds.build(
                ham_dir=tmp_path,
                subset_csv=tmp_path / "subset.csv",
                output_path=tmp_path / "out.jsonl",
                require_image=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# run_dermagent_subset
# ─────────────────────────────────────────────────────────────────────────────


def _write_subset_jsonl(path: Path, n: int = 5) -> None:
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "case_id": f"ISIC_{i:07d}",
                "image_path": f"data/ham10000/images/ISIC_{i:07d}.jpg",
                "query": "What is the diagnosis for this skin lesion?",
                "ground_truth_label": classes[i % len(classes)],
                "patient_context": {"age": "55", "sex": "male", "localization": "back"},
            }) + "\n")


class TestLoadSubset:
    def test_reads_jsonl(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        _write_subset_jsonl(p, n=3)
        cases = rds.load_subset(p)
        assert len(cases) == 3
        assert cases[0]["case_id"] == "ISIC_0000000"

    def test_max_cases_caps_load(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        _write_subset_jsonl(p, n=10)
        cases = rds.load_subset(p, max_cases=4)
        assert len(cases) == 4

    def test_raises_when_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Build it first"):
            rds.load_subset(tmp_path / "nope.jsonl")


class TestMockRunner:
    def test_run_one_mock_returns_metrics_record(self):
        case = {
            "case_id": "ISIC_X",
            "ground_truth_label": "mel",
            "image_path": "x.jpg",
            "query": "?",
            "patient_context": {},
        }
        rec = rds._run_one_mock(case)
        assert rec["case_id"] == "ISIC_X"
        assert rec["predicted_label"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
        assert isinstance(rec["top3_predictions"], list)
        assert len(rec["top3_predictions"]) == 3
        assert 0.0 <= rec["consensus_score"] <= 1.0
        # confidence and per_class_probs included
        assert "confidence" in rec
        assert "per_class_probs" in rec
        assert abs(sum(rec["per_class_probs"].values()) - 1.0) < 1e-6

    def test_mock_is_deterministic(self):
        case = {"case_id": "fixed", "ground_truth_label": "nv",
                "image_path": "x.jpg", "query": "?", "patient_context": {}}
        a = rds._run_one_mock(case)
        b = rds._run_one_mock(case)
        assert a["predicted_label"] == b["predicted_label"]


class TestSummarise:
    def test_summary_metrics(self):
        records = [
            {"predicted_label": "mel", "ground_truth_label": "mel",
             "top3_predictions": ["mel", "nv", "bcc"],
             "early_exit": True, "debate_rounds": 0, "tool_calls": 2,
             "total_tokens": 100, "latency_s": 0.5},
            {"predicted_label": "nv", "ground_truth_label": "mel",
             "top3_predictions": ["nv", "mel", "bkl"],
             "early_exit": False, "debate_rounds": 2, "tool_calls": 4,
             "total_tokens": 200, "latency_s": 1.0},
        ]
        s = rds._summarise(records)
        assert s["n_cases"] == 2
        assert s["accuracy"] == 0.5            # 1 of 2 top1 correct
        assert s["top3_accuracy"] == 1.0       # both have GT in top3
        assert s["early_exit_rate"] == 0.5
        assert s["avg_tool_calls"] == 3.0
        assert s["avg_debate_rounds"] == 1.0

    def test_empty(self):
        s = rds._summarise([])
        assert s == {"n_cases": 0}


class TestRunnerEnd2End:
    def test_mock_run_produces_outputs(self, tmp_path: Path):
        subset = tmp_path / "subset.jsonl"
        _write_subset_jsonl(subset, n=7)
        out_dir = tmp_path / "results"

        rc = rds.main([
            "--mock",
            "--subset", str(subset),
            "--output-dir", str(out_dir),
        ])
        assert rc == 0

        # One per-case JSONL + one metrics JSON, both timestamped.
        jsonl = list(out_dir.glob("dermagent_subset_mock_*.jsonl"))
        metrics = list(out_dir.glob("dermagent_subset_mock_*.metrics.json"))
        assert len(jsonl) == 1
        assert len(metrics) == 1

        lines = jsonl[0].read_text().splitlines()
        assert len(lines) == 7

        summary = json.loads(metrics[0].read_text())
        assert summary["mode"] == "mock"
        assert summary["n_cases"] == 7
        assert "accuracy" in summary

    def test_real_mode_currently_blocked(self, tmp_path: Path):
        subset = tmp_path / "subset.jsonl"
        _write_subset_jsonl(subset, n=2)
        rc = rds.main([
            "--real",
            "--subset", str(subset),
            "--output-dir", str(tmp_path / "results"),
        ])
        # Returns 3 (NotImplementedError) until factory layer lands.
        assert rc == 3

    def test_requires_mode_flag(self):
        with pytest.raises(SystemExit):
            rds._build_argparser().parse_args([])
