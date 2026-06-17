"""Curate a DermAbench v1-lite subset + clinician annotation worksheet (B3).

Two modes:

    build   Take one or more source DermAbench JSONL files (SCIN, DDI, …),
            fairness-stratified-sample ~N cases (balanced across Fitzpatrick
            skin types so the dark-skin subgroup has enough N for a CI, and
            diverse across conditions), write:
              * <out>.jsonl          — curated gold cases (status=pending)
              * <out>_worksheet.csv  — clinician review sheet (Phase B3)

    apply   Read a clinician-filled worksheet + the curated JSONL, merge the
            clinician fields (reference differential, management, malignancy,
            approval) back into each case, set annotation_status="frozen",
            and write the frozen gold set used by DermAbenchScorer.

Fairness-aware sampling: real SCIN is heavily light-skin (II=1007 vs VI=27).
Proportional sampling would leave ~1 dark-skin case — useless for a fairness
gap. We instead give each present Fitzpatrick group an equal target quota
(capped at availability), redistributing the slack from small groups, then
round-robin across distinct conditions within each group for diversity.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("curate_dermabench")

# Clinician worksheet columns (B3). Auto-* are pre-filled; the rest are blank
# for the dermatologist to complete. image_url is a clickable public link so
# the reviewer can view each lesion without downloading the image set.
WORKSHEET_FIELDS = [
    "case_id", "source", "fitzpatrick_type", "image_url",
    "clinical_history", "auto_diagnosis", "auto_icd10", "auto_management",
    # ── clinician fills below ──
    "ref_dx_1", "ref_dx_2", "ref_dx_3",
    "management",       # biopsy | monitor | reassure
    "is_malignant",     # Y | N
    "approve",          # Y | N  (N = exclude from frozen set)
    "notes",
]

# Public HTTPS base for SCIN images (no auth) so worksheet links open in a
# browser / Excel / Sheets directly.
_SCIN_IMG_BASE = "https://storage.googleapis.com/dx-scin-public-data/dataset/images/"


def _image_url(case: dict[str, Any]) -> str:
    """Derive a clickable public image URL from a case's image_path.

    SCIN paths embed 'dataset/images/<id>.png' → map to the public bucket
    URL. Other sources fall back to the raw path (clinician resolves locally).
    """
    path = case.get("image_path", "") or ""
    marker = "dataset/images/"
    if case.get("source") == "scin" and marker in path:
        return _SCIN_IMG_BASE + path.split(marker, 1)[1]
    return path


# ── IO ──────────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(cases: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            n += 1
    return n


# ── Fairness-aware stratified curation ──────────────────────────────────────
def curate(
    cases: list[dict[str, Any]],
    target_n: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select ~target_n cases balanced across Fitzpatrick groups and diverse
    across conditions. Cases with no Fitzpatrick are pooled under '?' and
    contribute only if quota remains."""
    rng = random.Random(seed)

    by_fitz: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        fitz = (c.get("fitzpatrick_type") or "?").strip() or "?"
        by_fitz[fitz].append(c)

    # Equal target per known Fitzpatrick group, capped at availability.
    known = [g for g in by_fitz if g != "?"]
    if not known:
        known = list(by_fitz)
    per_group = max(1, target_n // max(1, len(known)))

    selected: list[dict[str, Any]] = []
    leftover_capacity = 0
    # First pass: equal quota per group (condition round-robin within group).
    for g in known:
        pool = by_fitz[g]
        take = min(per_group, len(pool))
        selected.extend(_diverse_by_condition(pool, take, rng))
        leftover_capacity += max(0, per_group - len(pool))

    # Second pass: redistribute slack from small groups (e.g. VI) to the
    # largest groups so we still reach ~target_n.
    if leftover_capacity > 0:
        chosen_ids = {c["case_id"] for c in selected}
        big_first = sorted(known, key=lambda g: -len(by_fitz[g]))
        for g in big_first:
            if leftover_capacity <= 0:
                break
            remaining = [c for c in by_fitz[g] if c["case_id"] not in chosen_ids]
            extra = _diverse_by_condition(remaining, min(leftover_capacity, len(remaining)), rng)
            selected.extend(extra)
            chosen_ids.update(c["case_id"] for c in extra)
            leftover_capacity -= len(extra)

    rng.shuffle(selected)
    return selected[:target_n]


def _diverse_by_condition(
    pool: list[dict[str, Any]], take: int, rng: random.Random,
) -> list[dict[str, Any]]:
    """Round-robin across distinct diagnosis labels to maximise condition
    diversity within a quota."""
    if take >= len(pool):
        return list(pool)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in pool:
        buckets[c["ground_truth"].get("diagnosis_label", "?")].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    order = list(buckets.keys())
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    while len(out) < take:
        progressed = False
        for label in order:
            if buckets[label]:
                out.append(buckets[label].pop())
                progressed = True
                if len(out) >= take:
                    break
        if not progressed:
            break
    return out


# ── Worksheet ────────────────────────────────────────────────────────────────
def write_worksheet(cases: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=WORKSHEET_FIELDS)
        w.writeheader()
        for c in cases:
            gt = c.get("ground_truth", {})
            w.writerow({
                "case_id": c["case_id"],
                "source": c.get("source", ""),
                "fitzpatrick_type": c.get("fitzpatrick_type", ""),
                "image_url": _image_url(c),
                "clinical_history": c.get("clinical_history", ""),
                "auto_diagnosis": gt.get("diagnosis_label", ""),
                "auto_icd10": gt.get("icd10_code") or "",
                "auto_management": gt.get("management") or "",
                # clinician-blank
                "ref_dx_1": "", "ref_dx_2": "", "ref_dx_3": "",
                "management": "", "is_malignant": "", "approve": "", "notes": "",
            })


# ── Apply filled worksheet → frozen gold ─────────────────────────────────────
def apply_worksheet(
    cases: list[dict[str, Any]], worksheet_path: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge clinician annotations back into the curated cases and freeze
    the approved ones. Returns (frozen_cases, approved, rejected)."""
    by_id = {c["case_id"]: c for c in cases}
    with worksheet_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    frozen: list[dict[str, Any]] = []
    approved = rejected = 0
    for row in rows:
        cid = (row.get("case_id") or "").strip()
        case = by_id.get(cid)
        if case is None:
            continue
        if (row.get("approve") or "").strip().upper() not in ("Y", "YES", "1"):
            rejected += 1
            continue
        ref = [row.get("ref_dx_1", ""), row.get("ref_dx_2", ""), row.get("ref_dx_3", "")]
        ref = [r.strip() for r in ref if r and r.strip()]
        gt = case.setdefault("ground_truth", {})
        if ref:
            gt["reference_differential"] = ref
        mgmt = (row.get("management") or "").strip().lower()
        if mgmt:
            gt["management"] = mgmt
        mal = (row.get("is_malignant") or "").strip().upper()
        if mal in ("Y", "YES", "1"):
            gt["is_malignant"] = True
        elif mal in ("N", "NO", "0"):
            gt["is_malignant"] = False
        case["annotation_status"] = "frozen"
        case["annotator"] = "abdurrahim"
        if (row.get("notes") or "").strip():
            case["clinician_notes"] = row["notes"].strip()
        frozen.append(case)
        approved += 1
    return frozen, approved, rejected


# ── CLI ──────────────────────────────────────────────────────────────────────
def _summarise(cases: list[dict[str, Any]]) -> None:
    from collections import Counter
    fitz = Counter(c.get("fitzpatrick_type", "?") or "?" for c in cases)
    dx = Counter(c["ground_truth"].get("diagnosis_label", "?") for c in cases)
    coded = sum(1 for c in cases if c["ground_truth"].get("icd10_code"))
    print("\n" + "=" * 60)
    print(f" DermAbench v1-lite curation — {len(cases)} cases")
    print("=" * 60)
    print(f"  Fitzpatrick: {dict(sorted(fitz.items()))}")
    print(f"  Distinct conditions: {len(dx)}")
    print(f"  ICD-coded: {coded}/{len(cases)} ({100*coded//max(1,len(cases))}%)")
    print(f"  Top conditions: {dict(dx.most_common(8))}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Curate DermAbench v1-lite + worksheet.")
    sub = p.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("build", help="source JSONL(s) → curated subset + worksheet")
    b.add_argument("--sources", nargs="+", required=True,
                   help="One or more DermAbench source JSONL files.")
    b.add_argument("--target-n", type=int, default=200)
    b.add_argument("--out", default="data/dermabench/dermabench_v1lite.jsonl")
    b.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("apply", help="filled worksheet + subset JSONL → frozen gold")
    a.add_argument("--subset", required=True, help="The curated v1-lite JSONL.")
    a.add_argument("--worksheet", required=True, help="Clinician-filled CSV.")
    a.add_argument("--out", default="data/dermabench/dermabench_v1lite_frozen.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.mode == "build":
        cases: list[dict[str, Any]] = []
        for src in args.sources:
            loaded = _load_jsonl(Path(src))
            logger.info("Loaded %d cases from %s", len(loaded), src)
            cases.extend(loaded)
        curated = curate(cases, target_n=args.target_n, seed=args.seed)
        out = Path(args.out)
        _write_jsonl(curated, out)
        ws = out.with_name(out.stem + "_worksheet.csv")
        write_worksheet(curated, ws)
        _summarise(curated)
        print(f"  Gold (pending):  {out}  ({len(curated)} cases)")
        print(f"  Worksheet (B3):  {ws}\n")
        return 0

    if args.mode == "apply":
        cases = _load_jsonl(Path(args.subset))
        frozen, approved, rejected = apply_worksheet(cases, Path(args.worksheet))
        out = Path(args.out)
        _write_jsonl(frozen, out)
        print(f"\n  Frozen gold: {out}")
        print(f"  Approved: {approved}  |  Rejected: {rejected}\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
