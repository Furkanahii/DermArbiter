"""Tests for the DermAbench v1-lite curation + clinician worksheet (B3).

Offline; uses the synthetic builder to fabricate a skewed-Fitzpatrick
source so the fairness-aware sampling is exercised.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import sys
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import build_dermabench as bdb       # noqa: E402
import curate_dermabench as cur      # noqa: E402


def _skewed_source(n_per_fitz: dict[str, int]) -> list[dict]:
    """Fabricate DermAbench cases with a given Fitzpatrick skew."""
    base = bdb.build_synthetic(sum(n_per_fitz.values()) + 7, seed=1)
    out = []
    idx = 0
    for fitz, cnt in n_per_fitz.items():
        for _ in range(cnt):
            c = dict(base[idx % len(base)])
            c = json.loads(json.dumps(c))          # deep copy
            c["case_id"] = f"C-{fitz}-{idx}"
            c["fitzpatrick_type"] = fitz
            idx += 1
            out.append(c)
    return out


class TestCurate:
    def test_fairness_oversamples_rare_dark_skin(self):
        # Heavy light-skin skew (mirrors real SCIN: VI very rare).
        src = _skewed_source({"I": 300, "II": 300, "III": 200,
                              "IV": 100, "V": 50, "VI": 10})
        curated = cur.curate(src, target_n=120, seed=42)
        fitz = Counter(c["fitzpatrick_type"] for c in curated)
        # All 10 VI cases should be taken (not ~2 a proportional sample gives).
        assert fitz["VI"] == 10
        # Light and dark groups should be far closer than the 30:1 source skew.
        assert fitz["I"] <= 4 * fitz["VI"]

    def test_target_size_respected(self):
        src = _skewed_source({"I": 100, "II": 100, "III": 100,
                              "IV": 100, "V": 100, "VI": 100})
        curated = cur.curate(src, target_n=120, seed=1)
        assert len(curated) <= 120
        assert len(curated) >= 100   # should get close to target

    def test_condition_diversity(self):
        src = _skewed_source({"III": 200, "IV": 200})
        curated = cur.curate(src, target_n=60, seed=3)
        # synthetic builder cycles 7 classes → diverse conditions present
        classes = {c["ground_truth"]["diagnosis_class"] for c in curated}
        assert len(classes) >= 5

    def test_worksheet_has_clinician_blanks(self, tmp_path):
        src = _skewed_source({"III": 20, "V": 20})
        curated = cur.curate(src, target_n=20, seed=2)
        ws = tmp_path / "ws.csv"
        cur.write_worksheet(curated, ws)
        rows = list(csv.DictReader(ws.open(encoding="utf-8-sig")))
        assert len(rows) == len(curated)
        # auto fields filled, clinician fields blank
        assert rows[0]["auto_diagnosis"] != ""
        for blank in ("ref_dx_1", "management", "is_malignant", "approve", "notes"):
            assert rows[0][blank] == ""


class TestApply:
    def _curated_and_worksheet(self, tmp_path):
        src = _skewed_source({"III": 10, "V": 10})
        curated = cur.curate(src, target_n=10, seed=5)
        sub = tmp_path / "sub.jsonl"
        cur._write_jsonl(curated, sub)
        ws = tmp_path / "ws.csv"
        cur.write_worksheet(curated, ws)
        return curated, sub, ws

    def test_apply_freezes_approved_only(self, tmp_path):
        curated, sub, ws = self._curated_and_worksheet(tmp_path)
        rows = list(csv.DictReader(ws.open(encoding="utf-8-sig")))
        for i, r in enumerate(rows):
            r["ref_dx_1"] = "melanoma"; r["ref_dx_2"] = "nevus"
            r["management"] = "biopsy"; r["is_malignant"] = "Y"
            r["approve"] = "Y" if i < 3 else "N"
        with ws.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

        cases = cur._load_jsonl(sub)
        frozen, approved, rejected = cur.apply_worksheet(cases, ws)
        assert approved == 3
        assert rejected == len(rows) - 3
        assert all(c["annotation_status"] == "frozen" for c in frozen)
        assert all(c["annotator"] == "abdurrahim" for c in frozen)

    def test_apply_merges_clinician_fields(self, tmp_path):
        curated, sub, ws = self._curated_and_worksheet(tmp_path)
        rows = list(csv.DictReader(ws.open(encoding="utf-8-sig")))
        rows[0]["ref_dx_1"] = "melanoma"; rows[0]["ref_dx_2"] = "atypical nevus"
        rows[0]["management"] = "biopsy"; rows[0]["is_malignant"] = "Y"
        rows[0]["approve"] = "Y"; rows[0]["notes"] = "urgent excision"
        with ws.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

        cases = cur._load_jsonl(sub)
        frozen, _, _ = cur.apply_worksheet(cases, ws)
        c = frozen[0]
        assert c["ground_truth"]["reference_differential"] == ["melanoma", "atypical nevus"]
        assert c["ground_truth"]["management"] == "biopsy"
        assert c["ground_truth"]["is_malignant"] is True
        assert c["clinician_notes"] == "urgent excision"

    def test_frozen_set_is_scoreable(self, tmp_path):
        # A frozen curated case should drop straight into the scorer.
        from dermarbiter.evaluation.dermabench import DermAbenchScorer
        curated, sub, ws = self._curated_and_worksheet(tmp_path)
        rows = list(csv.DictReader(ws.open(encoding="utf-8-sig")))
        for r in rows:
            r["ref_dx_1"] = r["auto_diagnosis"]; r["management"] = "monitor"
            r["is_malignant"] = "N"; r["approve"] = "Y"
        with ws.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        frozen, _, _ = cur.apply_worksheet(cur._load_jsonl(sub), ws)
        preds = [{
            "case_id": c["case_id"],
            "predicted_label": c["ground_truth"]["diagnosis_class"] or "x",
            "top3_predictions": [c["ground_truth"]["diagnosis_class"] or "x"],
            "consensus_score": 0.8, "predicted_icd10": c["ground_truth"].get("icd10_code"),
            "predicted_snomed": c["ground_truth"].get("snomed_code"),
            "reasoning": "x", "cited_cards": ["EC-1"],
            "urgent_referral_flag": False, "recommended_management": "monitor",
        } for c in frozen]
        sc = DermAbenchScorer(frozen, preds)
        r = sc.score_all()
        assert r["n_cases"] == len(frozen)
