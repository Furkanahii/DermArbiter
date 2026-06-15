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


# ── Real-source loaders (framework stubs — fill as data lands) ──────────────
def load_source(source: str, raw_dir: Path) -> list[dict[str, Any]]:
    """Dispatch to a real-source loader. Each must return unified-schema
    dicts with annotation_status='pending' (clinician completes in B3).

    These intentionally raise until the corresponding raw data is staged,
    so a half-configured run fails loudly rather than emitting empty data.
    """
    raise NotImplementedError(
        f"Real-source loader for '{source}' not yet wired. Stage the raw "
        f"{source} files under {raw_dir} and implement the parser. "
        f"Schema: see module docstring. Use --source synthetic to test the "
        f"harness offline in the meantime."
    )


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
