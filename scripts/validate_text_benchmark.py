"""Validate the proposed primary text-only dataset contract without model calls.

Usage: python scripts/validate_text_benchmark.py cases.jsonl --output audit.json
This structural preflight cannot establish clinical validity, text provenance,
absence of semantic answer leakage, or absence of model training contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTEXT_FIELDS = {"age", "sex", "location", "duration", "symptoms", "history"}
TEXT_ORIGINS = {"native_clinical_text", "clinician_authored_vignette"}
REQUIRED_TEXT = (
    "case_id", "source_id", "split", "modality", "text_origin",
    "clinical_history", "query", "annotation_status", "reviewer_id",
    "license", "source_url",
)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def model_input(case: dict[str, Any]) -> dict[str, Any]:
    """Whitelist model-visible fields; never serialize a gold case wholesale.

    Source IDs, paths, labels, reviewer metadata, and splits stay evaluator-side.
    The caller must run validate_cases first. Text itself still requires review.
    """
    context = case.get("patient_context", {})
    return {
        "clinical_history": case["clinical_history"],
        "query": case["query"],
        "patient_context": {k: v for k, v in context.items() if k in CONTEXT_FIELDS},
    }


def validate_cases(cases: list[Any]) -> dict[str, Any]:
    errors: list[str] = []
    ids: set[str] = set()
    source_splits: dict[str, str] = {}
    history_splits: dict[str, str] = {}
    counts: dict[str, int] = {}
    if not cases:
        errors.append("Dataset is empty.")
    for index, case in enumerate(cases, 1):
        prefix = f"row {index}"
        if not isinstance(case, dict):
            errors.append(f"{prefix}: expected an object")
            continue
        for field in REQUIRED_TEXT:
            if not _nonempty(case.get(field)):
                errors.append(f"{prefix}: {field} must be a non-empty string")
        cid = case.get("case_id")
        if _nonempty(cid):
            cid = cid.strip()
            if cid in ids:
                errors.append(f"{prefix}: duplicate case_id {cid}")
            ids.add(cid)
        split = case.get("split")
        valid_split = isinstance(split, str) and split in {"dev", "test"}
        if not valid_split:
            errors.append(f"{prefix}: split must be dev or test")
        else:
            counts[split] = counts.get(split, 0) + 1
        if case.get("modality") != "text":
            errors.append(f"{prefix}: modality must be text")
        origin = case.get("text_origin")
        if not isinstance(origin, str) or origin not in TEXT_ORIGINS:
            errors.append(f"{prefix}: primary track requires native or clinician-authored text")
        for field in ("image_path", "image_url", "images", "image_embeddings"):
            if case.get(field):
                errors.append(f"{prefix}: {field} is forbidden in the primary text track")
        if case.get("annotation_status") != "frozen":
            errors.append(f"{prefix}: annotation_status must be frozen after clinical review")
        if case.get("reviewer_id") in ("auto", "synthetic"):
            errors.append(f"{prefix}: automated annotation is not clinical review")
        if case.get("source") == "synthetic":
            errors.append(f"{prefix}: synthetic fixtures are not primary benchmark evidence")
        gold = case.get("ground_truth")
        if not isinstance(gold, dict) or not _nonempty(gold.get("diagnosis_label")):
            errors.append(f"{prefix}: ground_truth.diagnosis_label is required")
        context = case.get("patient_context", {})
        if not isinstance(context, dict):
            errors.append(f"{prefix}: patient_context must be an object")
        else:
            for key, value in context.items():
                if key not in CONTEXT_FIELDS:
                    errors.append(f"{prefix}: unapproved patient_context field {key}")
                if value is not None and not isinstance(value, (str, int, float)):
                    errors.append(f"{prefix}: patient_context.{key} must be scalar")
        if valid_split:
            source = case.get("source_id")
            if _nonempty(source):
                source = source.strip()
                previous = source_splits.setdefault(source, split)
                if previous != split:
                    errors.append(f"{prefix}: source_id crosses dev/test: {source}")
            history = case.get("clinical_history")
            if _nonempty(history):
                normalized = " ".join(history.casefold().split())
                previous = history_splits.setdefault(normalized, split)
                if previous != split:
                    errors.append(f"{prefix}: duplicate clinical history crosses dev/test")
    return {
        "schema_version": "text-preflight-v1",
        "n_cases": len(cases),
        "split_counts": counts,
        "structural_checks_passed": not errors,
        "errors": errors,
        "limitations": [
            "Provenance, license terms and clinical review declarations are not independently verified.",
            "Manual answer-leakage review and near-duplicate/source overlap audit are still required.",
            "This tool validates a proposed input contract; existing runners do not consume it yet.",
        ],
    }


def audit_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    cases = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
    report = validate_cases(cases)
    report["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_file(args.dataset)
    except (OSError, UnicodeError, ValueError) as exc:
        report = {"structural_checks_passed": False, "errors": [str(exc)]}
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
