"""Tests for DermAbench: code lookup, builder, and 8-dimension scorer.

All offline, no network/GPU. Uses the synthetic builder as a fixture so
the scorer is exercised end-to-end on self-consistent gold cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from types import SimpleNamespace

from dermarbiter.evaluation import derm_codes as dc
from dermarbiter.evaluation.dermabench import (
    DermAbenchScorer, state_to_dermabench_prediction,
)

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

    def test_extended_common_conditions(self):
        # SCIN-majority everyday conditions get codes via the extended table.
        assert dc.icd10_for("Eczema") == "L20.9"
        assert dc.icd10_for("Psoriasis") == "L40.9"
        assert dc.icd10_for("Tinea") == "B35.9"
        assert dc.snomed_for("Urticaria") == "126485001"
        rec = dc.reference_record("Eczema")
        assert rec["icd10_code"] == "L20.9"
        assert rec["is_malignant"] is False
        assert rec["management"] == "monitor"
        assert rec["diagnosis_class"] == ""   # not a HAM 7-class


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

    def test_missing_raw_file_raises(self):
        # Real loader on an empty dir → FileNotFoundError (fail loud).
        with pytest.raises(FileNotFoundError):
            bdb.load_source("ddi", Path("/tmp/dermabench_nowhere_xyz"))

    def test_pubmed_still_stub(self):
        with pytest.raises(NotImplementedError):
            bdb.load_source("pubmed", Path("/tmp/nowhere"))

    def test_unknown_source(self):
        with pytest.raises(SystemExit):
            bdb.load_source("nonexistent", Path("/tmp"))

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

    # noqa
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


# ─────────────────────────────────────────────────────────────────────────────
# state_to_dermabench_prediction — BlackboardState → prediction bridge
# ─────────────────────────────────────────────────────────────────────────────
class TestStateBridge:
    def _state(self, **over):
        spec = SimpleNamespace(
            top3_differential=["melanoma", "nevus", "seborrheic keratosis"],
            cited_cards=["EC-a1", "EC-b2"],
        )
        gen = SimpleNamespace(top3_differential=["melanoma"],
                              cited_cards=["EC-b2", "EC-c3"])
        base = dict(
            final_diagnosis=["melanoma", "melanocytic nevus", "seborrheic keratosis"],
            consensus_score=0.78,
            clinical_report="Enlarging asymmetric lesion; melanoma favored.",
            final_icd10_mappings={"melanoma": "C43.9"},
            final_snomed_mappings={},
            briefs={"specialist": spec, "generalist": gen},
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_label_normalised(self):
        rec = state_to_dermabench_prediction("c1", self._state())
        assert rec["predicted_label"] == "mel"
        assert rec["top3_predictions"] == ["mel", "nv", "bkl"]

    def test_icd_from_mapping_snomed_from_fallback(self):
        rec = state_to_dermabench_prediction("c1", self._state())
        assert rec["predicted_icd10"] == "C43.9"        # from moderator mapping
        assert rec["predicted_snomed"] == "372244006"   # fallback reference table

    def test_cited_cards_union(self):
        rec = state_to_dermabench_prediction("c1", self._state())
        assert rec["cited_cards"] == ["EC-a1", "EC-b2", "EC-c3"]

    def test_triage_proxy_malignant_urgent(self):
        rec = state_to_dermabench_prediction("c1", self._state())
        assert rec["urgent_referral_flag"] is True
        assert rec["recommended_management"] == "biopsy"

    def test_benign_not_urgent(self):
        st = self._state(final_diagnosis=["melanocytic nevus"],
                         final_icd10_mappings={})
        rec = state_to_dermabench_prediction("c1", st)
        assert rec["predicted_label"] == "nv"
        assert rec["urgent_referral_flag"] is False
        assert rec["recommended_management"] == "reassure"

    def test_dict_state_accepted(self):
        st = {"final_diagnosis": ["melanoma"], "consensus_score": 0.5,
              "clinical_report": "x", "final_icd10_mappings": {},
              "final_snomed_mappings": {}, "briefs": {}}
        rec = state_to_dermabench_prediction("c1", st)
        assert rec["predicted_label"] == "mel"
        assert rec["cited_cards"] == []

    def test_moderator_fallback_when_consensus_empty(self):
        mod = SimpleNamespace(top3_differential=["basal cell carcinoma", "nv"],
                              cited_cards=[])
        st = self._state(final_diagnosis=[], briefs={"moderator": mod})
        rec = state_to_dermabench_prediction("c1", st)
        assert rec["predicted_label"] == "bcc"

    def test_fitz_and_latency_optional(self):
        rec = state_to_dermabench_prediction("c1", self._state(),
                                             fitzpatrick_type="V", latency_s=3.14159)
        assert rec["fitzpatrick_type"] == "V"
        assert rec["latency_s"] == 3.142

    def test_bridge_output_scoreable(self):
        # The bridge output should be directly consumable by the scorer.
        gold = [{
            "case_id": "c1", "fitzpatrick_type": "IV",
            "ground_truth": {
                "diagnosis_label": "mel", "icd10_code": "C43.9",
                "snomed_code": "372244006", "is_malignant": True,
                "management": "biopsy", "reference_differential": ["mel", "nv"],
                "history_key_features": ["enlarging", "asymmetric"],
            },
        }]
        pred = [state_to_dermabench_prediction("c1", self._state())]
        sc = DermAbenchScorer(gold, pred)
        r = sc.score_all()
        assert r["n_cases"] == 1
        assert r["dimensions"]["1_visual_diagnosis"] == 1.0  # mel == mel
        assert r["dimensions"]["3_coding"] == 1.0            # both codes match


# ─────────────────────────────────────────────────────────────────────────────
# Real-source loaders (fixture CSVs mimic each dataset's real columns)
# ─────────────────────────────────────────────────────────────────────────────
class TestRealLoaders:
    def test_ddi_loader(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "ddi_metadata.csv").write_text(
            "DDI_file,skin_tone,malignant,disease\n"
            "000001.png,12,False,melanocytic nevus\n"
            "000002.png,56,True,melanoma\n"
            "000003.png,34,False,seborrheic keratosis\n",
            encoding="utf-8",
        )
        cases = bdb.load_ddi(tmp_path)
        assert len(cases) == 3
        c0, c1, c2 = cases
        # Fitzpatrick mapping 12→II(light), 56→V(dark), 34→III
        assert c0["fitzpatrick_type"] == "II"
        assert c1["fitzpatrick_type"] == "V"
        assert c2["fitzpatrick_type"] == "III"
        # DDI malignancy flag overrides code-table guess
        assert c1["ground_truth"]["is_malignant"] is True
        assert c0["ground_truth"]["is_malignant"] is False
        # auto code enrichment
        assert c1["ground_truth"]["icd10_code"] == "C43.9"   # melanoma
        # pending until clinician review
        assert all(c["annotation_status"] == "pending" for c in cases)
        assert c1["source"] == "ddi"

    def test_derm1m_loader(self, tmp_path):
        (tmp_path / "Derm1M_v2_pretrain.csv").write_text(
            "filename,disease_label,truncated_caption,age,gender,body_location\n"
            "IIYI/0_1.png,melanoma,Enlarging dark lesion on the back,60,male,back\n"
            "edu/3_2.png,nevus,No age information,No age information,female,arm\n",
            encoding="utf-8",
        )
        cases = bdb.load_derm1m(tmp_path)
        assert len(cases) == 2
        assert cases[0]["clinical_history"] == "Enlarging dark lesion on the back"
        assert cases[0]["patient_context"]["age"] == "60"
        # "No age information" sentinel cleaned to empty
        assert cases[1]["clinical_history"] == ""
        assert cases[1]["patient_context"]["age"] == ""
        assert cases[0]["ground_truth"]["icd10_code"] == "C43.9"

    def test_scin_loader_real_schema(self, tmp_path):
        # Real SCIN: two files joined on case_id, one-hot body/symptom cols,
        # list/dict label strings, FSTn fitzpatrick in the labels file.
        (tmp_path / "scin_cases.csv").write_text(
            "case_id,age_group,sex_at_birth,body_parts_arm,body_parts_leg,"
            "condition_symptoms_itching,condition_symptoms_pain,"
            "condition_duration,image_1_path\n"
            "C1,50-59,MALE,YES,,YES,,ONE_WEEK,dataset/images/aaa.png\n"
            "C2,20-29,FEMALE,,YES,,YES,ONE_MONTH,dataset/images/bbb.png\n"
            "C3,30-39,MALE,YES,,,,,dataset/images/ccc.png\n",  # no label → skipped
            encoding="utf-8",
        )
        (tmp_path / "scin_labels.csv").write_text(
            "case_id,dermatologist_skin_condition_on_label_name,"
            "weighted_skin_condition_label,dermatologist_fitzpatrick_skin_type_label_1\n"
            'C1,"[\'Eczema\', \'Psoriasis\']","{\'Eczema\': 0.7, \'Psoriasis\': 0.3}",FST5\n'
            'C2,"[\'Melanoma\']","{\'Melanoma\': 0.9}",FST2\n'
            "C3,,,FST3\n",   # no label
            encoding="utf-8",
        )
        cases = bdb.load_scin(tmp_path)
        # C3 has no dermatologist label → skipped
        assert len(cases) == 2
        c1, c2 = cases
        # weighted dict → highest weight wins
        assert c1["ground_truth"]["diagnosis_label"] == "Eczema"
        assert c1["fitzpatrick_type"] == "V"            # FST5
        assert c2["fitzpatrick_type"] == "II"           # FST2
        # one-hot reconstruction into narrative
        assert "arm" in c1["clinical_history"]
        assert "itching" in c1["clinical_history"]
        assert "duration: one week" in c1["clinical_history"]
        # Eczema enriched via extended code table
        assert c1["ground_truth"]["icd10_code"] == "L20.9"
        # Melanoma maps to HAM class + code
        assert c2["ground_truth"]["diagnosis_class"] == "mel"
        assert c2["ground_truth"]["icd10_code"] == "C43.9"
        # Silver differential seeded from the weighted multi-reader label,
        # ordered by descending weight (Eczema 0.7 before Psoriasis 0.3).
        assert c1["ground_truth"]["reference_differential"] == ["Eczema", "Psoriasis"]
        assert c2["ground_truth"]["reference_differential"] == ["Melanoma"]
        # Dataset-derived → silver, not pending; clinician B3 upgrades later.
        assert all(c["annotation_status"] == "silver_scin" for c in cases)
        assert all(c["annotator"] == "scin_dataset" for c in cases)

    def test_loaders_emit_scoreable_schema(self, tmp_path):
        # A DDI case should drop straight into the scorer's gold contract.
        (tmp_path / "images").mkdir()
        (tmp_path / "ddi_metadata.csv").write_text(
            "DDI_file,skin_tone,malignant,disease\n"
            "x.png,56,True,melanoma\n", encoding="utf-8",
        )
        gold = bdb.load_ddi(tmp_path)
        # mark frozen + add clinician fields so the scorer can run
        gold[0]["ground_truth"]["reference_differential"] = ["mel", "nv"]
        pred = [{
            "case_id": gold[0]["case_id"],
            "predicted_label": "mel", "top3_predictions": ["mel"],
            "consensus_score": 0.9, "predicted_icd10": "C43.9",
            "predicted_snomed": "372244006", "reasoning": "x",
            "cited_cards": ["EC-1"], "urgent_referral_flag": True,
            "recommended_management": "biopsy",
        }]
        from dermarbiter.evaluation.dermabench import DermAbenchScorer
        sc = DermAbenchScorer(gold, pred)
        r = sc.score_all()
        assert r["n_cases"] == 1
        assert r["dimensions"]["1_visual_diagnosis"] == 1.0
        assert r["dimensions"]["7_safety"] == 1.0   # malignant + urgent flagged
