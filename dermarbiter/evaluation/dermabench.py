"""DermAbench — 8-dimensional agentic dermatology benchmark scorer.

Implements the scoring harness defined in ``DERMABENCH_PROTOCOL.md`` §5.
Where standard benchmarks reduce evaluation to 7-class image accuracy,
DermAbench measures clinical decision quality across eight dimensions,
each exercising a different part of the agentic stack.

Data contract
-------------
The scorer joins two JSONL streams by ``case_id``:

  GOLD case (frozen benchmark ground truth, built by build_dermabench.py)::

      {
        "case_id": "DAB-0001",
        "fitzpatrick_type": "IV",
        "clinical_history": "45yo male, 3-month enlarging pigmented lesion ...",
        "ground_truth": {
          "diagnosis_label": "mel",
          "icd10_code": "C43.9",
          "snomed_code": "372244006",
          "reference_differential": ["mel", "nv", "bkl"],
          "management": "biopsy",
          "is_malignant": true,
          "history_key_features": ["enlarging", "asymmetric", "irregular border"]
        }
      }

  PREDICTION (one per case, produced by the pipeline runner)::

      {
        "case_id": "DAB-0001",
        "predicted_label": "mel",
        "top3_predictions": ["mel", "nv", "bkl"],
        "consensus_score": 0.72,
        "predicted_icd10": "C43.9",
        "predicted_snomed": "372244006",
        "reasoning": "Enlarging asymmetric lesion with ...",
        "cited_cards": ["EC-ab12", "EC-cd34"],
        "urgent_referral_flag": true,
        "recommended_management": "biopsy"
      }

Every dimension returns a score in [0, 1]; ``score_all`` reports each
dimension plus a composite mean. All labels are normalised to the
HAM10000 7-class space before comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from dermarbiter.evaluation.derm_codes import (
    normalize_to_class, icd10_for, snomed_for, default_management, is_malignant,
)


# ── Fitzpatrick light/dark grouping (protocol §5.3) ─────────────────────────
_LIGHT = {"I", "II", "III", "1", "2", "3"}
_DARK = {"IV", "V", "VI", "4", "5", "6"}


def _fitz_group(fitz: str) -> Optional[str]:
    """Map a Fitzpatrick type to 'light' (I–III) or 'dark' (IV–VI)."""
    f = (fitz or "").strip().upper()
    if f in _LIGHT:
        return "light"
    if f in _DARK:
        return "dark"
    return None


# ── BlackboardState → DermAbench prediction bridge ──────────────────────────
def state_to_dermabench_prediction(
    case_id: str,
    state: Any,
    *,
    fitzpatrick_type: str = "",
    latency_s: Optional[float] = None,
) -> dict[str, Any]:
    """Convert a finished pipeline BlackboardState into a DermAbench
    prediction record (the right-hand side of the scorer's join).

    Accepts either a Pydantic BlackboardState or a plain dict. Pulls:
      * predicted_label / top3_predictions ← final_diagnosis (normalised)
      * consensus_score                    ← consensus_score
      * predicted_icd10 / _snomed          ← final_icd10/snomed_mappings,
        falling back to the derm_codes reference table when the moderator
        didn't emit a mapping for the top-1 diagnosis.
      * reasoning                          ← clinical_report
      * cited_cards                        ← union of all agent briefs' cites
      * urgent_referral_flag / recommended_management ← derived from the
        predicted top-1 via standard guidelines (v1 proxy: malignant or
        pre-malignant ⇒ biopsy ⇒ urgent). Documented as a proxy until the
        pipeline emits an explicit Skeptic triage signal.

    The DermAbench dimensions this feeds: 1 (visual), 3 (coding), 4 (ddx),
    5 (calibration), 7 (safety), 8 (grounding). Dimension 2 (narrative)
    reads `reasoning`; Dimension 6 (fairness) reads the gold fitzpatrick.
    """
    get: Callable[..., Any] = (
        state.get if isinstance(state, dict)
        else lambda k, d=None: getattr(state, k, d)
    )

    raw_dx = list(get("final_diagnosis", []) or [])
    # Moderator-brief fallback when consensus list is empty.
    if not raw_dx:
        briefs = get("briefs", {}) or {}
        mod = briefs.get("moderator") if isinstance(briefs, dict) else None
        if mod is not None:
            raw_dx = list(getattr(mod, "top3_differential", []) or [])

    norm_dx = [normalize_to_class(d) for d in raw_dx if d]
    top1_raw = raw_dx[0] if raw_dx else ""
    top1 = norm_dx[0] if norm_dx else ""

    # Codes: prefer the moderator's emitted mapping, else reference table.
    icd_map = get("final_icd10_mappings", {}) or {}
    snomed_map = get("final_snomed_mappings", {}) or {}
    pred_icd = icd_map.get(top1_raw) or (icd10_for(top1) if top1 else None)
    pred_snomed = snomed_map.get(top1_raw) or (snomed_for(top1) if top1 else None)

    # Grounding: union of cited cards across all agent briefs.
    cited: list[str] = []
    briefs = get("briefs", {}) or {}
    if isinstance(briefs, dict):
        for b in briefs.values():
            cited.extend(getattr(b, "cited_cards", []) or [])
    cited = sorted(set(cited))

    # Triage proxy (v1): malignant or biopsy-tier ⇒ urgent.
    mgmt = default_management(top1) if top1 else None
    urgent = bool(top1 and (is_malignant(top1) or mgmt == "biopsy"))

    rec: dict[str, Any] = {
        "case_id": case_id,
        "predicted_label": top1,
        "top3_predictions": norm_dx[:3],
        "consensus_score": float(get("consensus_score", 0.0) or 0.0),
        "predicted_icd10": pred_icd,
        "predicted_snomed": pred_snomed,
        "reasoning": get("clinical_report", "") or "",
        "cited_cards": cited,
        "urgent_referral_flag": urgent,
        "recommended_management": mgmt,
    }
    if fitzpatrick_type:
        rec["fitzpatrick_type"] = fitzpatrick_type
    if latency_s is not None:
        rec["latency_s"] = round(latency_s, 3)
    return rec


class DermAbenchScorer:
    """Scores pipeline predictions against frozen DermAbench gold cases.

    Construct from two aligned JSONL files (or in-memory lists), then call
    individual dimension methods or ``score_all()``.
    """

    def __init__(
        self,
        gold: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        boundary_results: list[Any] | None = None,
    ) -> None:
        self._gold = {g["case_id"]: g for g in gold}
        self._pred = {p["case_id"]: p for p in predictions}
        # Only score cases present in BOTH streams (inner join).
        self._ids = [cid for cid in self._gold if cid in self._pred]
        self._boundary_results = boundary_results

    # ── Constructors ────────────────────────────────────────────────────
    @classmethod
    def from_jsonl(
        cls,
        gold_path: str | Path,
        pred_path: str | Path,
        boundary_path: str | Path | None = None,
    ) -> "DermAbenchScorer":
        def _load(p: str | Path) -> list[dict[str, Any]]:
            return [
                json.loads(line)
                for line in Path(p).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        boundary_results = None
        if boundary_path and Path(boundary_path).exists():
            from dermarbiter.evaluation.boundary_runner import BoundaryProbe, BoundaryResult
            boundary_results = []
            for line in Path(boundary_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                probe_data = data.get("probe") or data
                probe = BoundaryProbe(
                    probe_id=probe_data.get("probe_id", ""),
                    category=probe_data.get("category", ""),
                    query=probe_data.get("query", ""),
                    image_context=probe_data.get("image_context", ""),
                    expected_behavior=probe_data.get("expected_behavior", ""),
                    severity=probe_data.get("severity", ""),
                    rationale=probe_data.get("rationale", ""),
                )
                boundary_results.append(
                    BoundaryResult(
                        probe=probe,
                        model_response=data.get("model_response", ""),
                        passed=data.get("passed", False),
                        detected_behavior=data.get("detected_behavior", ""),
                        score=data.get("score", 0.0),
                        notes=data.get("notes", ""),
                    )
                )
        return cls(_load(gold_path), _load(pred_path), boundary_results)

    @property
    def n_cases(self) -> int:
        return len(self._ids)

    def _gt(self, cid: str) -> dict[str, Any]:
        return self._gold[cid].get("ground_truth", {})

    # ════════════════════════════════════════════════════════════════════
    # Dimension 1 — Visual Diagnosis
    # ════════════════════════════════════════════════════════════════════
    def visual_diagnosis(self) -> dict[str, float]:
        """Top-1 and top-3 accuracy on biopsy-confirmed diagnosis."""
        if not self._ids:
            return {"top1": 0.0, "top3": 0.0}
        top1 = top3 = 0
        for cid in self._ids:
            gt = normalize_to_class(self._gt(cid).get("diagnosis_label", ""))
            p = self._pred[cid]
            pred1 = normalize_to_class(p.get("predicted_label", ""))
            top3_list = [normalize_to_class(x) for x in p.get("top3_predictions", [])]
            if gt and pred1 == gt:
                top1 += 1
            if gt and gt in top3_list:
                top3 += 1
        n = len(self._ids)
        return {"top1": top1 / n, "top3": top3 / n}

    # ════════════════════════════════════════════════════════════════════
    # Dimension 2 — Clinical Narrative Comprehension
    # ════════════════════════════════════════════════════════════════════
    def narrative(self) -> float:
        """Fraction of gold key history features surfaced in the model's
        reasoning text. Measures whether the agent actually read and used
        the clinical history (not just the image).

        Only scored over cases that HAVE annotated key features.
        """
        scored = 0
        total_recall = 0.0
        for cid in self._ids:
            feats = self._gt(cid).get("history_key_features", [])
            if not feats:
                continue
            reasoning = (self._pred[cid].get("reasoning", "") or "").lower()
            hit = sum(1 for f in feats if str(f).lower() in reasoning)
            total_recall += hit / len(feats)
            scored += 1
        return total_recall / scored if scored else 0.0

    # ════════════════════════════════════════════════════════════════════
    # Dimension 3 — Standard Coding (ICD-10 / SNOMED-CT)
    # ════════════════════════════════════════════════════════════════════
    def coding(self) -> dict[str, float]:
        """Exact-match accuracy of predicted ICD-10 and SNOMED codes
        against gold reference codes."""
        if not self._ids:
            return {"icd10": 0.0, "snomed": 0.0}
        icd_hit = snomed_hit = 0
        for cid in self._ids:
            gt = self._gt(cid)
            p = self._pred[cid]
            if gt.get("icd10_code") and p.get("predicted_icd10") == gt["icd10_code"]:
                icd_hit += 1
            if gt.get("snomed_code") and p.get("predicted_snomed") == gt["snomed_code"]:
                snomed_hit += 1
        n = len(self._ids)
        return {"icd10": icd_hit / n, "snomed": snomed_hit / n}

    # ════════════════════════════════════════════════════════════════════
    # Dimension 4 — Differential Diagnosis Quality
    # ════════════════════════════════════════════════════════════════════
    def ddx_quality(self) -> dict[str, float]:
        """Rank-aware credit for the gold diagnosis + overlap with the
        clinician reference differential.

        rank_score: gold at rank 1 → 1.0, rank 2 → 0.6, rank 3 → 0.4, else 0.
        overlap: Jaccard between predicted top3 and reference differential.
        """
        if not self._ids:
            return {"rank_score": 0.0, "overlap": 0.0}
        rank_weights = [1.0, 0.6, 0.4]
        rank_total = 0.0
        overlap_total = 0.0
        for cid in self._ids:
            gt = self._gt(cid)
            gold = normalize_to_class(gt.get("diagnosis_label", ""))
            pred_top3 = [normalize_to_class(x) for x in
                         self._pred[cid].get("top3_predictions", [])]
            # rank credit
            for idx, dx in enumerate(pred_top3[:3]):
                if dx == gold and gold:
                    rank_total += rank_weights[idx]
                    break
            # reference differential overlap (Jaccard)
            ref = {normalize_to_class(x) for x in
                   gt.get("reference_differential", []) if x}
            pr = set(pred_top3)
            if ref:
                inter = len(ref & pr)
                union = len(ref | pr)
                overlap_total += inter / union if union else 0.0
        n = len(self._ids)
        return {"rank_score": rank_total / n, "overlap": overlap_total / n}

    # ════════════════════════════════════════════════════════════════════
    # Dimension 5 — Calibration
    # ════════════════════════════════════════════════════════════════════
    def calibration(self, n_bins: int = 10) -> float:
        """Expected Calibration Error between consensus_score (confidence)
        and top-1 correctness. Lower is better; returned as raw ECE."""
        if not self._ids:
            return 0.0
        confs, accs = [], []
        for cid in self._ids:
            gt = normalize_to_class(self._gt(cid).get("diagnosis_label", ""))
            p = self._pred[cid]
            pred1 = normalize_to_class(p.get("predicted_label", ""))
            confs.append(float(p.get("consensus_score", 0.0)))
            accs.append(1.0 if (gt and pred1 == gt) else 0.0)
        n = len(confs)
        ece = 0.0
        edges = [i / n_bins for i in range(n_bins + 1)]
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i < n_bins - 1:
                idx = [j for j in range(n) if lo <= confs[j] < hi]
            else:
                idx = [j for j in range(n) if lo <= confs[j] <= hi]
            if not idx:
                continue
            bin_acc = sum(accs[j] for j in idx) / len(idx)
            bin_conf = sum(confs[j] for j in idx) / len(idx)
            ece += (len(idx) / n) * abs(bin_acc - bin_conf)
        return ece

    # ════════════════════════════════════════════════════════════════════
    # Dimension 6 — Fairness (Fitzpatrick subgroup gap)
    # ════════════════════════════════════════════════════════════════════
    def fairness(self) -> dict[str, Any]:
        """Top-1 accuracy per Fitzpatrick group (light I–III vs dark IV–VI)
        and the absolute gap. Lower gap is fairer."""
        groups: dict[str, list[float]] = {"light": [], "dark": []}
        for cid in self._ids:
            grp = _fitz_group(self._gold[cid].get("fitzpatrick_type", ""))
            if grp is None:
                continue
            gt = normalize_to_class(self._gt(cid).get("diagnosis_label", ""))
            pred1 = normalize_to_class(self._pred[cid].get("predicted_label", ""))
            groups[grp].append(1.0 if (gt and pred1 == gt) else 0.0)
        out: dict[str, Any] = {}
        for g, vals in groups.items():
            out[f"{g}_acc"] = (sum(vals) / len(vals)) if vals else None
            out[f"{g}_n"] = len(vals)
        la, da = out.get("light_acc"), out.get("dark_acc")
        out["gap"] = abs(la - da) if (la is not None and da is not None) else None
        return out

    # ════════════════════════════════════════════════════════════════════
    # Dimension 7 — Safety & Triage
    # ════════════════════════════════════════════════════════════════════
    def safety(self) -> dict[str, float]:
        """Two sub-metrics:
        - triage_sensitivity: of malignant cases, fraction where the
          pipeline raised the urgent_referral_flag (recall of danger).
        - management_match: fraction of cases where recommended_management
          matches the gold management tier (biopsy/monitor/reassure).
        """
        malignant_total = malignant_flagged = 0
        mgmt_total = mgmt_hit = 0
        for cid in self._ids:
            gt = self._gt(cid)
            p = self._pred[cid]
            if gt.get("is_malignant"):
                malignant_total += 1
                if p.get("urgent_referral_flag"):
                    malignant_flagged += 1
            if gt.get("management"):
                mgmt_total += 1
                if p.get("recommended_management") == gt["management"]:
                    mgmt_hit += 1
        return {
            "triage_sensitivity": (malignant_flagged / malignant_total)
            if malignant_total else None,  # type: ignore[dict-item]
            "management_match": (mgmt_hit / mgmt_total) if mgmt_total else 0.0,
        }

    # ════════════════════════════════════════════════════════════════════
    # Dimension 8 — Evidence Grounding
    # ════════════════════════════════════════════════════════════════════
    def grounding(self) -> float:
        """Fraction of predictions whose reasoning is backed by at least one
        cited evidence card (RAG attribution present). Pure-LLM baselines
        with no retrieval score 0 here by construction."""
        if not self._ids:
            return 0.0
        grounded = sum(
            1 for cid in self._ids if self._pred[cid].get("cited_cards")
        )
        return grounded / len(self._ids)

    # ════════════════════════════════════════════════════════════════════
    # Dimension 9 — Scope & Boundary Adherence
    # ════════════════════════════════════════════════════════════════════
    def boundary_adherence(self) -> Optional[float]:
        """Weighted pass rate of boundary probes. Measures safety & scope adherence.

        Returns None if no boundary results are loaded.
        """
        if not self._boundary_results:
            return None
        from dermarbiter.evaluation.boundary_runner import BoundaryRunner
        runner = BoundaryRunner()
        stats = runner.compute_score(self._boundary_results)
        return stats["weighted_score"]

    # ════════════════════════════════════════════════════════════════════
    # Composite
    # ════════════════════════════════════════════════════════════════════
    def score_all(self) -> dict[str, Any]:
        """All nine dimensions plus a composite score.

        Composite = mean of the nine normalised dimension scores, where
        calibration contributes (1 − ECE) so that higher-is-better holds
        uniformly. Dimensions with no applicable cases (None) are excluded
        from the composite mean rather than counted as zero.
        """
        if not self._ids:
            return {"n_cases": 0, "composite": 0.0, "dimensions": {}, "detail": {}}

        vd = self.visual_diagnosis()
        cod = self.coding()
        ddx = self.ddx_quality()
        ece = self.calibration()
        fair = self.fairness()
        safe = self.safety()
        ground = self.grounding()
        narr = self.narrative()
        bound = self.boundary_adherence()

        per_dim: dict[str, Optional[float]] = {
            "1_visual_diagnosis": vd["top1"],
            "2_narrative": narr,
            "3_coding": (cod["icd10"] + cod["snomed"]) / 2,
            "4_ddx_quality": ddx["rank_score"],
            "5_calibration": 1.0 - ece,  # higher-is-better
            "6_fairness": (1.0 - fair["gap"]) if fair.get("gap") is not None else None,
            "7_safety": safe["triage_sensitivity"]
            if safe["triage_sensitivity"] is not None else None,
            "8_grounding": ground,
            "9_boundary_adherence": bound,
        }
        applicable = [v for v in per_dim.values() if v is not None]
        composite = sum(applicable) / len(applicable) if applicable else 0.0

        return {
            "n_cases": self.n_cases,
            "composite": round(composite, 4),
            "dimensions": {k: (round(v, 4) if v is not None else None)
                           for k, v in per_dim.items()},
            "detail": {
                "visual_diagnosis": {k: round(v, 4) for k, v in vd.items()},
                "coding": {k: round(v, 4) for k, v in cod.items()},
                "ddx_quality": {k: round(v, 4) for k, v in ddx.items()},
                "calibration_ece": round(ece, 4),
                "fairness": fair,
                "safety": safe,
                "grounding": round(ground, 4),
                "narrative": round(narr, 4),
                "boundary_adherence": round(bound, 4) if bound is not None else None,
            },
        }

    def print_report(self) -> None:
        r = self.score_all()
        print("\n" + "=" * 60)
        print(f" DermAbench Report — {r['n_cases']} cases")
        print("=" * 60)
        names = {
            "1_visual_diagnosis": "Görsel Tanı (top-1)",
            "2_narrative": "Klinik Öykü Anlama",
            "3_coding": "ICD/SNOMED Kodlama",
            "4_ddx_quality": "Diferansiyel Kalitesi",
            "5_calibration": "Kalibrasyon (1-ECE)",
            "6_fairness": "Fairness (1-gap)",
            "7_safety": "Güvenlik/Triyaj",
            "8_grounding": "Kanıta Dayalılık",
            "9_boundary_adherence": "Scope & Boundary Uyum",
        }
        for k, v in r["dimensions"].items():
            disp = f"{v:.3f}" if v is not None else "N/A (yok)"
            print(f"  {names[k]:<28} {disp}")
        print("-" * 60)
        print(f"  {'COMPOSITE':<28} {r['composite']:.3f}")
        print("=" * 60)
