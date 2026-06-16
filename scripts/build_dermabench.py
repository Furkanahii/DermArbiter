"""Build the DermAbench gold-standard case set (Phase B2).

Assembles multi-dimensional dermatology cases from public sources into the
unified DermAbench JSONL schema consumed by
``dermarbiter.evaluation.dermabench.DermAbenchScorer``.

Per the protocol (DERMABENCH_PROTOCOL.md), each gold case carries enough
ground truth to score all 8 dimensions:

    {
      "case_id": "DAB-0001",
      "source": "scin|ddi|derm1m|pubmed|synthetic",
      "image_path": "data/dermabench/images/DAB-0001.jpg",
      "fitzpatrick_type": "IV",
      "clinical_history": "45yo male, 3-month enlarging pigmented lesion ...",
      "query": "What is the most likely diagnosis and management?",
      "patient_context": {"age": "45", "sex": "male", "localization": "back"},
      "ground_truth": {
        "diagnosis_label": "mel",
        "diagnosis_class": "mel",
        "icd10_code": "C43.9",          # auto-enriched from derm_codes
        "snomed_code": "372244006",     # auto-enriched
        "is_malignant": true,           # auto-enriched
        "management": "biopsy",         # auto-enriched (clinician may override)
        "reference_differential": ["mel", "nv", "bkl"],   # clinician (B3)
        "history_key_features": ["enlarging", "asymmetric"] # clinician (B3)
      },
      "annotation_status": "pending|frozen",
      "annotator": "auto|abdurrahim"
    }

Modes
-----
    --source synthetic --n 60
        Generate a balanced synthetic fixture (testable offline, no data
        download). Useful for validating the scorer + harness end-to-end.

    --source scin | ddi | derm1m | pubmed
        Real-source loaders. SCIN/DDI/PubMed need their raw files staged
        under --raw-dir first (see download_datasets.py / manual steps).
        These are framework stubs that the team fills as data lands;
        they all emit the same unified schema + auto ICD/SNOMED enrichment.

Workflow
--------
1. Build raw cases (this script) → annotation_status="pending".
2. Dr. Yılmaz blind-reviews → adds reference_differential, management,
   history_key_features; sets annotation_status="frozen" (Phase B3).
3. Freeze + register before any evaluation runs (pre-registration).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

# Allow running as a plain script.
import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dermarbiter.evaluation.derm_codes import reference_record, all_classes

logger = logging.getLogger("build_dermabench")

DEFAULT_OUT = "data/dermabench/dermabench_v1.jsonl"


# ── ICD/SNOMED enrichment ───────────────────────────────────────────────────
def enrich_ground_truth(diagnosis_label: str,
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a ground_truth block from a diagnosis label + clinician extras.

    Auto-fills icd10/snomed/is_malignant/management from the reference code
    table; clinician-provided fields (reference_differential,
    history_key_features, management override) come via ``extra``.
    """
    ref = reference_record(diagnosis_label)
    gt: dict[str, Any] = {
        "diagnosis_label": diagnosis_label,
        "diagnosis_class": ref.get("diagnosis_class", ""),
        "icd10_code": ref.get("icd10_code"),
        "snomed_code": ref.get("snomed_code"),
        "is_malignant": ref.get("is_malignant", False),
        "management": ref.get("management"),
        "reference_differential": [],
        "history_key_features": [],
    }
    if extra:
        # Clinician overrides take precedence (e.g. management after review).
        gt.update({k: v for k, v in extra.items() if v is not None})
    return gt


# ── Synthetic fixture generator (offline-testable) ──────────────────────────
_SYN_HISTORY = {
    "mel": ("{age}yo {sex}, {months}-month history of an enlarging, "
            "asymmetric pigmented lesion on the {loc} with recent colour change.",
            ["enlarging", "asymmetric", "colour change"]),
    "nv": ("{age}yo {sex}, long-standing stable symmetric brown macule on the "
           "{loc}, no recent change.",
           ["stable", "symmetric", "no recent change"]),
    "bkl": ("{age}yo {sex}, waxy 'stuck-on' verrucous plaque on the {loc} "
            "present for years.",
            ["waxy", "stuck-on", "verrucous"]),
    "bcc": ("{age}yo {sex}, slowly growing pearly papule with telangiectasia "
            "on the sun-exposed {loc}.",
            ["pearly", "telangiectasia", "slow-growing"]),
    "akiec": ("{age}yo {sex}, rough scaly erythematous patch on chronically "
              "sun-damaged {loc}.",
              ["scaly", "erythematous", "sun-damaged"]),
    "df": ("{age}yo {sex}, firm dermal nodule on the {loc} with positive "
           "dimple sign.",
           ["firm", "dermal nodule", "dimple sign"]),
    "vasc": ("{age}yo {sex}, well-circumscribed red-purple vascular papule on "
             "the {loc} that blanches on pressure.",
             ["vascular", "red-purple", "blanches"]),
}
_LOCS = ["back", "face", "trunk", "lower leg", "scalp", "forearm"]
_FITZ = ["I", "II", "III", "IV", "V", "VI"]


def build_synthetic(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Class-balanced synthetic gold cases for scorer/harness validation.

    NOT for paper results — these have templated histories and no real
    images. Purpose: exercise the 8-dimension scorer offline.
    """
    rng = random.Random(seed)
    classes = all_classes()
    cases: list[dict[str, Any]] = []
    for i in range(n):
        cls = classes[i % len(classes)]
        tmpl, feats = _SYN_HISTORY[cls]
        age = rng.randint(25, 80)
        sex = rng.choice(["male", "female"])
        loc = rng.choice(_LOCS)
        fitz = rng.choice(_FITZ)
        months = rng.randint(2, 18)
        history = tmpl.format(age=age, sex=sex, months=months, loc=loc)
        cid = f"DAB-SYN-{i:04d}"
        gt = enrich_ground_truth(cls, extra={
            "reference_differential": [cls] + rng.sample(
                [c for c in classes if c != cls], 2),
            "history_key_features": feats,
        })
        cases.append({
            "case_id": cid,
            "source": "synthetic",
            "image_path": f"data/dermabench/images/{cid}.jpg",
            "fitzpatrick_type": fitz,
            "clinical_history": history,
            "query": "What is the most likely diagnosis and recommended management?",
            "patient_context": {"age": str(age), "sex": sex, "localization": loc},
            "ground_truth": gt,
            "annotation_status": "frozen",   # synthetic = self-consistent
            "annotator": "auto",
        })
    return cases


# ── Real-source loaders ─────────────────────────────────────────────────────
# Each parses a staged raw dataset into the unified DermAbench schema with
# auto ICD/SNOMED enrichment. They emit annotation_status="pending" because
# reference_differential + history_key_features still need the clinician
# blind-review (Phase B3) before freezing. Source-provided fields (DDI's
# malignancy flag, Fitzpatrick) override the code-table defaults.


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv
    if not path.exists():
        raise FileNotFoundError(
            f"Expected source file not found: {path}. Stage the raw data "
            f"under the --raw-dir first."
        )
    # utf-8-sig strips any BOM the published CSVs ship with.
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _first(row: dict[str, str], *keys: str) -> str:
    """First non-empty value across alternate column names."""
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


# DDI skin_tone groups → a representative Fitzpatrick type. DDI annotates in
# pairs (12 = Fitz I–II, 34 = III–IV, 56 = V–VI); the dataset's primary
# fairness axis is light (12) vs dark (56). We store the raw group in
# patient_context and pick a representative type for the binary light/dark
# scorer (12→II light, 34→III light-boundary, 56→V dark).
_DDI_SKINTONE_TO_FITZ = {"12": "II", "34": "III", "56": "V"}


def load_ddi(raw_dir: Path) -> list[dict[str, Any]]:
    """Stanford Diverse Dermatology Images (DDI).

    Expects ``raw_dir/ddi_metadata.csv`` with columns:
        DDI_file, skin_tone (12/34/56), malignant (True/False), disease
    and images under ``raw_dir/images/``.

    DDI is the fairness anchor: it ships Fitzpatrick groups + a biopsy-
    confirmed malignancy flag, so Dimensions 6 (fairness) and 7 (safety)
    get real ground truth here.
    """
    rows = _read_csv(raw_dir / "ddi_metadata.csv")
    images = raw_dir / "images"
    cases: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        fname = _first(row, "DDI_file", "ddi_file", "filename")
        disease = _first(row, "disease", "diagnosis", "label")
        if not fname or not disease:
            continue
        skin_tone = _first(row, "skin_tone", "skin_tone_group")
        fitz = _DDI_SKINTONE_TO_FITZ.get(skin_tone, "")
        malignant_raw = _first(row, "malignant", "is_malignant").lower()
        src_malignant = malignant_raw in ("true", "1", "yes")
        cid = f"DAB-DDI-{i:04d}"
        gt = enrich_ground_truth(disease, extra={
            # DDI's biopsy-confirmed malignancy overrides the code-table guess.
            "is_malignant": src_malignant,
        })
        cases.append({
            "case_id": cid,
            "source": "ddi",
            "image_path": str(images / fname),
            "fitzpatrick_type": fitz,
            "clinical_history": "",   # DDI has no narrative — B3 may add context
            "query": "What is the most likely diagnosis and recommended management?",
            "patient_context": {"skin_tone_group": skin_tone},
            "ground_truth": gt,
            "annotation_status": "pending",
            "annotator": "auto",
        })
    return cases


def load_derm1m(raw_dir: Path) -> list[dict[str, Any]]:
    """Derm1M manifest → DermAbench cases.

    Reuses the manifest schema parsed by build_case_rag_index: columns
    ``filename, disease_label, source, body_location, age, gender`` plus a
    caption/clinical-text column used as the narrative. Expects
    ``raw_dir/Derm1M_v2_pretrain.csv`` (or pass --raw-dir to its folder).

    Derm1M is the narrative anchor: its captions give a real clinical
    description for Dimension 2.
    """
    # find a manifest csv in raw_dir
    candidates = list(raw_dir.glob("Derm1M*.csv")) or list(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Derm1M manifest CSV under {raw_dir}")
    rows = _read_csv(candidates[0])
    cases: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        fname = _first(row, "filename")
        diagnosis = _first(row, "disease_label")
        if not fname or not diagnosis:
            continue
        caption = _first(row, "truncated_caption", "caption", "text", "clinical_text")
        # Skip Derm1M's "No <thing> information" sentinels.
        def _clean(v: str) -> str:
            return "" if (v.lower().startswith("no ") and "information" in v.lower()) else v
        cid = f"DAB-D1M-{i:04d}"
        gt = enrich_ground_truth(diagnosis)
        cases.append({
            "case_id": cid,
            "source": "derm1m",
            "image_path": str(raw_dir / fname),
            "fitzpatrick_type": "",   # Derm1M has no Fitzpatrick labels
            "clinical_history": _clean(caption),
            "query": "What is the most likely diagnosis and recommended management?",
            "patient_context": {
                "age": _clean(_first(row, "age")),
                "sex": _clean(_first(row, "gender", "sex")),
                "localization": _clean(_first(row, "body_location")),
            },
            "ground_truth": gt,
            "annotation_status": "pending",
            "annotator": "auto",
        })
    return cases


# SCIN dermatologist Fitzpatrick columns → representative type.
def _scin_fitz(row: dict[str, str]) -> str:
    val = _first(row,
                 "dermatologist_fitzpatrick_skin_type_label_1",
                 "fitzpatrick_skin_type", "fitzpatrick")
    # SCIN encodes e.g. "FST3" or "3"; extract the digit → Roman.
    digits = "".join(c for c in val if c.isdigit())
    roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
    return roman.get(digits[:1], "") if digits else ""


def load_scin(raw_dir: Path) -> list[dict[str, Any]]:
    """Google SCIN (Skin Condition Image Network).

    Expects ``raw_dir/scin_cases.csv`` (+ optional ``scin_labels.csv`` merged
    on case_id). Pulls the dermatologist condition label, Fitzpatrick,
    body part, symptoms, and demographics — rich metadata for the narrative
    and fairness dimensions.

    SCIN's label column is a weighted list; we take the top-weighted
    dermatologist label as the ground-truth diagnosis. (Cases without a
    confident dermatologist label should be filtered in B3.)
    """
    rows = _read_csv(raw_dir / "scin_cases.csv")
    cases: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        cid_src = _first(row, "case_id", "caseId", "id") or str(i)
        diagnosis = _first(
            row,
            "dermatologist_skin_condition_on_label_name",
            "dermatologist_skin_condition_label",
            "weighted_skin_condition_label",
            "label",
        )
        # label may be a "cond:weight,cond:weight" list — take the first.
        if "," in diagnosis or ":" in diagnosis:
            diagnosis = diagnosis.split(",")[0].split(":")[0].strip()
        image_rel = _first(row, "image_path", "image_id", "image_1_path")
        if not diagnosis:
            continue
        body = _first(row, "body_parts", "body_part", "anatom_site")
        symptoms = _first(row, "symptoms", "condition_symptoms")
        history_bits = [b for b in (
            f"affected area: {body}" if body else "",
            f"symptoms: {symptoms}" if symptoms else "",
        ) if b]
        cid = f"DAB-SCIN-{i:04d}"
        gt = enrich_ground_truth(diagnosis)
        cases.append({
            "case_id": cid,
            "source": "scin",
            "image_path": str(raw_dir / "images" / image_rel) if image_rel else "",
            "fitzpatrick_type": _scin_fitz(row),
            "clinical_history": "; ".join(history_bits),
            "query": "What is the most likely diagnosis and recommended management?",
            "patient_context": {
                "age": _first(row, "age_group", "age"),
                "sex": _first(row, "sex_at_birth", "sex"),
                "localization": body,
                "scin_case_id": cid_src,
            },
            "ground_truth": gt,
            "annotation_status": "pending",
            "annotator": "auto",
        })
    return cases


def load_pubmed(raw_dir: Path) -> list[dict[str, Any]]:
    """PubMed dermatology case reports — complex atypical narratives with
    ICD/SNOMED-coded pathology.

    Left as a documented stub: case-report formats vary too widely to
    parse generically. The intended pipeline is (a) retrieve derm case
    reports via the PubMed E-utilities API, (b) extract the
    image + history + final pathology diagnosis, (c) map to codes. This
    requires a bespoke extraction step the team will build once the other
    three sources are validated.
    """
    raise NotImplementedError(
        "PubMed case-report loader is a documented stub — formats vary too "
        "widely for a generic parser. Use ddi / derm1m / scin for v1-lite; "
        "PubMed atypical-narrative cases are a v1-full / journal-revision "
        f"addition. (raw_dir was {raw_dir})"
    )


_SOURCE_LOADERS = {
    "ddi": load_ddi,
    "derm1m": load_derm1m,
    "scin": load_scin,
    "pubmed": load_pubmed,
}


def load_source(source: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Dispatch to a real-source loader. Each returns unified-schema dicts
    with annotation_status='pending' (clinician completes + freezes in B3).
    """
    loader = _SOURCE_LOADERS.get(source)
    if loader is None:
        raise SystemExit(f"Unknown source: {source}")
    return loader(raw_dir)


# ── IO ──────────────────────────────────────────────────────────────────────
def write_jsonl(cases: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    return n


def summarise(cases: list[dict[str, Any]]) -> None:
    from collections import Counter
    dist = Counter(c["ground_truth"]["diagnosis_class"] for c in cases)
    fitz = Counter(c.get("fitzpatrick_type", "?") for c in cases)
    frozen = sum(1 for c in cases if c.get("annotation_status") == "frozen")
    print("\n" + "=" * 60)
    print(f" DermAbench build — {len(cases)} cases")
    print("=" * 60)
    print(f"  Frozen (annotation complete): {frozen}/{len(cases)}")
    print(f"  Class distribution: {dict(sorted(dist.items()))}")
    print(f"  Fitzpatrick distribution: {dict(sorted(fitz.items()))}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build DermAbench gold case set.")
    p.add_argument("--source", required=True,
                   choices=["synthetic", "scin", "ddi", "derm1m", "pubmed"])
    p.add_argument("--n", type=int, default=60,
                   help="Synthetic only: number of cases to generate.")
    p.add_argument("--raw-dir", default="data/dermabench/raw",
                   help="Where staged raw source files live (real sources).")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if args.source == "synthetic":
        cases = build_synthetic(args.n, seed=args.seed)
    else:
        cases = load_source(args.source, Path(args.raw_dir))

    out = Path(args.out)
    written = write_jsonl(cases, out)
    summarise(cases)
    print(f"  Output: {out}  ({written} cases)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
