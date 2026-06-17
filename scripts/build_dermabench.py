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


_FST_TO_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}


def _scin_fitz(label_row: dict[str, str]) -> str:
    """SCIN dermatologist Fitzpatrick (labels file) 'FSTn' → Roman."""
    val = _first(label_row, "dermatologist_fitzpatrick_skin_type_label_1")
    digits = "".join(c for c in val if c.isdigit())
    return _FST_TO_ROMAN.get(digits[:1], "") if digits else ""


def _scin_top_diagnosis(label_row: dict[str, str]) -> str:
    """Highest-weight dermatologist condition from a SCIN labels row.

    weighted_skin_condition_label is a dict-string like
    "{'Eczema': 0.41, 'Irritant Contact Dermatitis': 0.18}". Fall back to
    the first element of the list-string dermatologist_skin_condition_on
    _label_name "['Eczema', ...]" when the weighted dict is absent.
    """
    import ast
    weighted = (label_row.get("weighted_skin_condition_label") or "").strip()
    if weighted:
        try:
            d = ast.literal_eval(weighted)
            if isinstance(d, dict) and d:
                return max(d.items(), key=lambda kv: kv[1])[0].strip()
        except (ValueError, SyntaxError):
            pass
    listed = (label_row.get("dermatologist_skin_condition_on_label_name") or "").strip()
    if listed:
        try:
            lst = ast.literal_eval(listed)
            if isinstance(lst, list) and lst:
                return str(lst[0]).strip()
        except (ValueError, SyntaxError):
            pass
    return ""


def _scin_differential(label_row: dict[str, str], top_n: int = 3) -> list[str]:
    """Top-N dermatologist conditions by weight from a SCIN labels row.

    SCIN aggregates several dermatologists' reads into a weighted dict; the
    higher-weight entries form a real (silver) differential, not a single
    label. Used to seed ground_truth.reference_differential so DermAbench's
    DDx dimension is scoreable before clinician B3 review. Ordered by
    descending weight; falls back to the list-string when no weighted dict.
    """
    import ast
    weighted = (label_row.get("weighted_skin_condition_label") or "").strip()
    if weighted:
        try:
            d = ast.literal_eval(weighted)
            if isinstance(d, dict) and d:
                ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
                return [str(k).strip() for k, _ in ranked[:top_n] if str(k).strip()]
        except (ValueError, SyntaxError):
            pass
    listed = (label_row.get("dermatologist_skin_condition_on_label_name") or "").strip()
    if listed:
        try:
            lst = ast.literal_eval(listed)
            if isinstance(lst, list):
                return [str(x).strip() for x in lst[:top_n] if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return []


def _scin_collect_onehot(row: dict[str, str], prefix: str) -> list[str]:
    """Collect human-readable tokens from SCIN one-hot columns.

    e.g. prefix='body_parts_' → ['arm', 'torso back'] for the columns
    body_parts_arm=YES, body_parts_torso_back=YES.
    """
    out = []
    for col, val in row.items():
        if col.startswith(prefix) and (val or "").strip().upper() == "YES":
            token = col[len(prefix):].replace("_", " ").strip()
            if token and token != "other":
                out.append(token)
    return out


def load_scin(raw_dir: Path) -> list[dict[str, Any]]:
    """Google SCIN (Skin Condition Image Network) — real two-file schema.

    Reads scin_cases.csv (demographics, one-hot body-parts/symptoms, image
    paths) and scin_labels.csv (dermatologist condition label + Fitzpatrick),
    joined on case_id. Builds a narrative from the structured fields.

    Note: SCIN labels are broad everyday-dermatology conditions (Eczema,
    Urticaria, Tinea, Psoriasis, …) — mostly OUTSIDE the HAM10000 7-class
    space. derm_codes enriches the overlap; the rest get empty ICD/SNOMED
    that the clinician completes in B3. This breadth is intentional:
    DermAbench tests holistic clinical competence, not just cancer-screening.

    Only cases with a confident dermatologist label are emitted.
    """
    cases_rows = _read_csv(raw_dir / "scin_cases.csv")
    label_rows = _read_csv(raw_dir / "scin_labels.csv")
    labels_by_id = {r.get("case_id", ""): r for r in label_rows}

    cases: list[dict[str, Any]] = []
    kept = 0
    for row in cases_rows:
        cid_src = (row.get("case_id") or "").strip()
        lab = labels_by_id.get(cid_src, {})
        diagnosis = _scin_top_diagnosis(lab)
        if not diagnosis:
            continue   # no confident dermatologist label → skip (B3 policy)
        image_rel = _first(row, "image_1_path", "image_path")
        body = _scin_collect_onehot(row, "body_parts_")
        symptoms = _scin_collect_onehot(row, "condition_symptoms_")
        duration = _first(row, "condition_duration").replace("_", " ").lower()
        history_bits = []
        if body:
            history_bits.append("affected area: " + ", ".join(body))
        if symptoms:
            history_bits.append("symptoms: " + ", ".join(symptoms))
        if duration and duration not in ("", "unknown"):
            history_bits.append("duration: " + duration)

        cid = f"DAB-SCIN-{kept:04d}"
        gt = enrich_ground_truth(diagnosis)
        # Seed a silver differential from SCIN's weighted multi-reader label
        # so the DDx dimension is scoreable pre-clinician. The B3 clinician
        # later overrides reference_differential and upgrades the status.
        gt["reference_differential"] = _scin_differential(lab)
        cases.append({
            "case_id": cid,
            "source": "scin",
            # image_1_path already begins "dataset/images/..." in SCIN.
            "image_path": str(raw_dir / image_rel) if image_rel else "",
            "fitzpatrick_type": _scin_fitz(lab),
            "clinical_history": "; ".join(history_bits),
            "query": "What is the most likely diagnosis and recommended management?",
            "patient_context": {
                "age": _first(row, "age_group"),
                "sex": _first(row, "sex_at_birth"),
                "localization": ", ".join(body),
                "scin_case_id": cid_src,
            },
            "ground_truth": gt,
            # Silver gold: dataset-derived labels, scoreable now; a clinician
            # later blind-reviews these (B3) and upgrades them to "frozen".
            "annotation_status": "silver_scin",
            "annotator": "scin_dataset",
        })
        kept += 1
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
