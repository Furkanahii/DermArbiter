"""Tests for DermAbench: code lookup, builder, and 8-dimension scorer.

All offline, no network/GPU. Uses the synthetic builder as a fixture so
the scorer is exercised end-to-end on self-consistent gold cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dermarbiter.evaluation import derm_codes as dc
from dermarbiter.evaluation.dermabench import DermAbenchScorer

import sys
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import build_dermabench as bdb  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# derm_codes lookup
# ─────────────────────────────────────────────────────────────────────────────
class TestDermCodes:
    def test_canonical_codes(self):
        assert dc.icd10_for("mel") == "C43.9"
        assert dc.icd10_for("nv") == "D22.9"
        assert dc.snomed_for("bcc") == "254701007"

    def test_freetext_normalisation(self):
        assert dc.icd10_for("Melanocytic Nevus") == "D22.9"
        assert dc.icd10_for("malignant melanoma") == "C43.9"
        assert dc.normalize_to_class("atypical seborrheic keratosis") == "bkl" \
            or dc.normalize_to_class("seborrheic keratosis") == "bkl"

    def test_malignancy_flags(self):
        assert dc.is_malignant("mel") is True
        assert dc.is_malignant("bcc") is True
        assert dc.is_malignant("nv") is False

    def test_management_tiers(self):
        assert dc.default_management("mel") == "biopsy"
        assert dc.default_management("nv") == "reassure"
        assert dc.default_management("akiec") == "biopsy"   # pre-malignant

    def test_unknown_passthrough(self):
        # Unknown label maps to itself (lowercased), no code.
        assert dc.icd10_for("not_a_real_disease") is None
        assert dc.reference_record("not_a_real_disease") == {}

    def test_reference_record_complete(self):
        rec = dc.reference_record("mel")
        assert rec["diagnosis_class"] == "mel"
        assert rec["icd10_code"] == "C43.9"
        assert rec["is_malignant"] is True
        assert rec["management"] == "biopsy"

    def test_all_classes(self):
        assert set(dc.all_classes()) == {
            "nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"}


# ─────────────────────────────────────────────────────────────────────────────
# build_dermabench
# ─────────────────────────────────────────────────────────────────────────────
class TestBuilder:
    def test_synthetic_balanced(self):
        cases = bdb.build_synthetic(70, seed=42)
        assert len(cases) == 70
        from collections import Counter
        dist = Counter(c["ground_truth"]["diagnosis_class"] for c in cases)
        # 7 classes, 70 cases → 10 each
        assert all(n == 10 for n in dist.values())
        assert len(dist) == 7

    def test_synthetic_schema(self):
        case = bdb.build_synthetic(7, seed=1)[0]
        for key in ("case_id", "source", "image_path", "fitzpatrick_type",
                    "clinical_history", "query", "patient_context",
                    "ground_truth", "annotation_status"):
            assert key in case
        gt = case["ground_truth"]
        for key in ("diagnosis_label", "diagnosis_class", "icd10_code",
                    "snomed_code", "is_malignant", "management",
                    "reference_differential", "history_key_features"):
            assert key in gt

    def test_enrichment_auto_codes(self):
        gt = bdb.enrich_ground_truth("melanoma")
        assert gt["icd10_code"] == "C43.9"
        assert gt["is_malignant"] is True

    def test_enrichment_clinician_override(self):
        gt = bdb.enrich_ground_truth("mel", extra={
            "management": "monitor",   # clinician overrides default biopsy
            "reference_differential": ["mel", "nv"],
        })
        assert gt["management"] == "monitor"
        assert gt["reference_differential"] == ["mel", "nv"]

    def test_real_source_raises_until_wired(self):
        with pytest.raises(NotImplementedError):
            bdb.load_source("scin", Path("/tmp/nowhere"))

    def test_write_jsonl(self, tmp_path):
        cases = bdb.build_synthetic(7)
        out = tmp_path / "dab.jsonl"
        n = bdb.write_jsonl(cases, out)
        assert n == 7
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 7
        assert json.loads(lines[0])["case_id"].startswith("DAB-SYN")


# ─────────────────────────────────────────────────────────────────────────────
# DermAbenchScorer — perfect predictions
# ─────────────────────────────────────────────────────────────────────────────
def _perfect_predictions(gold: list[dict]) -> list[dict]:
    """Build a prediction stream that should score ~1.0 on every dimension."""
    preds = []
    for g in gold:
        gt = g["ground_truth"]
        feats = " ".join(gt["history_key_features"])
        preds.append({
            "case_id": g["case_id"],
            "predicted_label": gt["diagnosis_class"],
            "top3_predictions": [gt["diagnosis_class"]] + [
                x for x in gt["reference_differential"]
                if x != gt["diagnosis_class"]][:2],
            "consensus_score": 1.0,
            "predicted_icd10": gt["icd10_code"],
            "predicted_snomed": gt["snomed_code"],
            "reasoning": f"Findings: {feats}.",
            "cited_cards": ["EC-x1"],
            "urgent_referral_flag": gt["is_malignant"],
            "recommended_management": gt["management"],
        })
    return preds


class TestScorerPerfect:
    @pytest.fixture
    def gold(self):
        return bdb.build_synthetic(70, seed=7)

    def test_visual_diagnosis_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        vd = sc.visual_diagnosis()
        assert vd["top1"] == 1.0
        assert vd["top3"] == 1.0

    def test_coding_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        cod = sc.coding()
        assert cod["icd10"] == 1.0
        assert cod["snomed"] == 1.0

    def test_narrative_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        assert sc.narrative() == pytest.approx(1.0)

    def test_grounding_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        assert sc.grounding() == 1.0

    def test_safety_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        safe = sc.safety()
        assert safe["triage_sensitivity"] == 1.0
        assert safe["management_match"] == 1.0

    def test_calibration_perfect_is_low_ece(self, gold):
        # conf=1.0 and always correct → ECE ≈ 0
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        assert sc.calibration() == pytest.approx(0.0, abs=1e-9)

    def test_fairness_perfect_zero_gap(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        fair = sc.fairness()
        assert fair["light_acc"] == 1.0
        assert fair["dark_acc"] == 1.0
        assert fair["gap"] == pytest.approx(0.0)

    def test_composite_perfect(self, gold):
        sc = DermAbenchScorer(gold, _perfect_predictions(gold))
        r = sc.score_all()
        assert r["composite"] == pytest.approx(1.0, abs=1e-6)
        assert r["n_cases"] == 70


# ─────────────────────────────────────────────────────────────────────────────
# DermAbenchScorer — edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestScorerEdges:
    def test_empty_streams(self):
        sc = DermAbenchScorer([], [])
        r = sc.score_all()
        assert r["n_cases"] == 0
        assert r["composite"] == 0.0

    def test_inner_join_only(self):
        gold = bdb.build_synthetic(7, seed=3)
        # predictions for only 3 of 7 cases
        preds = _perfect_predictions(gold)[:3]
        sc = DermAbenchScorer(gold, preds)
        assert sc.n_cases == 3

    def test_grounding_zero_for_pure_llm(self):
        # A baseline with no cited_cards → grounding 0 (by construction).
        gold = bdb.build_synthetic(7, seed=5)
        preds = _perfect_predictions(gold)
        for p in preds:
            p["cited_cards"] = []
        sc = DermAbenchScorer(gold, preds)
        assert sc.grounding() == 0.0

    def test_wrong_predictions_low_scores(self):
        gold = bdb.build_synthetic(14, seed=9)
        preds = []
        for g in gold:
            preds.append({
                "case_id": g["case_id"],
                "predicted_label": "wrong_label_xyz",
                "top3_predictions": ["wrong_label_xyz"],
                "consensus_score": 0.9,    # confident AND wrong → bad calibration
                "predicted_icd10": "X00.0",
                "predicted_snomed": "000",
                "reasoning": "no relevant features",
                "cited_cards": [],
                "urgent_referral_flag": False,
                "recommended_management": "reassure",
            })
        sc = DermAbenchScorer(gold, preds)
        vd = sc.visual_diagnosis()
        assert vd["top1"] == 0.0
        assert sc.coding()["icd10"] == 0.0
        # confident+wrong → ECE high → calibration dim low
        assert sc.calibration() > 0.5
