"""Builders for the text-only benchmark tracks.

Track A  ``scin_text``       SCIN patient self-reports -> closed-set differential
                             (``text-dx-v1``), gold = dermatologist weighted labels.
Track B  ``medmcqa_skin``    MedMCQA train, subject Skin, single answer (``text-mcq-pilot-v1``).
         ``medxpertqa_derm`` MedXpertQA Text test, body_system Integumentary.

Builders read local, hash-pinned source files; they never download, never invent
labels and never use answer explanations as model input. Output is
``pilot_candidate`` unless the caller declares the data frozen after review.
Splits are deterministic and grouped so identical model-visible text never
crosses dev/test.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dermarbiter.research.text_benchmark import DX_VERSION, VERSION, validate_cases

SCIN_URL = "https://github.com/google-research-datasets/scin"
MEDMCQA_URL = "https://huggingface.co/datasets/openlifescienceai/medmcqa/tree/{rev}"
MEDXPERTQA_URL = "https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/tree/{rev}"

# Questions that depend on a figure/photo cannot be answered from text alone.
IMAGE_DEPENDENT = re.compile(
    r"\b(image|images|picture|photo|photograph|figure|fig\.|diagram|"
    r"shown (below|above|here|in)|given (below|above)|depicted|illustrated)\b",
    re.IGNORECASE,
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def assign_split(group_key: str, dev_fraction: float, salt: str) -> str:
    """Deterministic hash split; the same group always lands in the same split."""
    if not 0 <= dev_fraction <= 1:
        raise ValueError("dev_fraction must be in [0, 1]")
    bucket = int(hashlib.sha256(f"{salt}:{group_key}".encode()).hexdigest()[:8], 16) % 10_000
    return "dev" if bucket < dev_fraction * 10_000 else "test"


def shuffled_order(n: int, key: str) -> list[int]:
    """Deterministic permutation of range(n) from a string key (no global RNG)."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{key}:{i}".encode()).hexdigest())


def _status_fields(data_status: str, accepted_by: str | None) -> dict[str, Any]:
    if data_status not in ("pilot_candidate", "frozen"):
        raise ValueError("data_status must be pilot_candidate or frozen")
    if data_status == "frozen" and not (accepted_by and accepted_by.strip()):
        raise ValueError("Declaring data frozen requires --accepted-by (who reviewed it)")
    out: dict[str, Any] = {"data_status": data_status}
    if data_status == "frozen":
        out["accepted_by"] = accepted_by.strip()
    return out


# ════════════════════════════════════════════════════════════════════════
# Track B — MedMCQA Skin (train split)
# ════════════════════════════════════════════════════════════════════════

def build_medmcqa_skin(
    rows: Iterable[dict[str, Any]],
    *,
    revision: str,
    source_sha256: str,
    dev_fraction: float = 0.2,
    salt: str = "dermarbiter-v1",
    data_status: str = "pilot_candidate",
    accepted_by: str | None = None,
    shuffle_options: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``shuffle_options`` permutes A-D per question (seeded by salt + id) and
    remaps the key, removing the source's answer-position skew (~35-40% "A")."""
    status = _status_fields(data_status, accepted_by)
    excluded: Counter[str] = Counter()
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("subject_name") != "Skin":
            continue
        if row.get("choice_type") != "single":
            excluded["multi_answer"] += 1
            continue
        try:
            cop = int(row.get("cop"))
        except (TypeError, ValueError):
            cop = -1
        if cop not in range(4):
            excluded["no_answer_key"] += 1
            continue
        options = [row.get(k) for k in ("opa", "opb", "opc", "opd")]
        if any(not isinstance(o, str) or not o.strip() for o in options) or \
                len({norm_text(o) for o in options}) < 4:
            excluded["empty_or_duplicate_options"] += 1
            continue
        q = row.get("question") or ""
        if not q.strip():
            excluded["empty_question"] += 1
            continue
        if IMAGE_DEPENDENT.search(q):
            excluded["image_dependent"] += 1
            continue
        by_question.setdefault(norm_text(q), []).append({**row, "_cop": cop, "_opts": options})

    cases = []
    for qkey, group in by_question.items():
        keys = {(tuple(sorted(norm_text(o) for o in g["_opts"])),
                 norm_text(g["_opts"][g["_cop"]])) for g in group}
        if len({k[1] for k in keys}) > 1:
            excluded["conflicting_duplicate_keys"] += len(group)
            continue
        excluded["exact_duplicates_merged"] += len(group) - 1
        row = group[0]
        cid = "medmcqa:" + str(row["id"])
        opts, key = [o.strip() for o in row["_opts"]], row["_cop"]
        order = list(range(4))
        if shuffle_options:
            order = shuffled_order(4, f"{salt}:options:{cid}")
        options = dict(zip("ABCD", [opts[i] for i in order]))
        answer = "ABCD"[order.index(key)]
        cases.append({
            "schema_version": VERSION, "case_id": cid, "source_id": cid,
            "source_url": MEDMCQA_URL.format(rev=revision), "license": "Apache-2.0 (dataset card)",
            "source_revision": revision, "source_file_sha256": source_sha256,
            "original_split": "train", "split": assign_split(qkey, dev_fraction, salt),
            "modality": "text", "text_origin": "published_exam_question",
            "gold_origin": "published_answer_key", **status,
            "question": row["question"].strip(),
            "options": options, "answer": answer,
            "metadata": {"topic_name": row.get("topic_name"),
                         "options_shuffled": shuffle_options,
                         "original_answer": "ABCD"[key]},
            "limitations": [
                "Public train split: may appear in model pre-training data (contamination).",
                "Published answer key not re-reviewed by a clinician.",
            ],
        })
    validate_cases(cases)
    report = _report("medmcqa_skin", cases, excluded,
                     source={"revision": revision, "sha256": source_sha256})
    report["options_shuffled"] = shuffle_options
    return cases, report


# ════════════════════════════════════════════════════════════════════════
# Track B — MedXpertQA (Integumentary)
# ════════════════════════════════════════════════════════════════════════

def build_medxpertqa_derm(
    rows: Iterable[dict[str, Any]],
    *,
    revision: str,
    source_sha256: str,
    data_status: str = "pilot_candidate",
    accepted_by: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All items go to ``test``: MedXpertQA is itself a held-out test set."""
    status = _status_fields(data_status, accepted_by)
    excluded: Counter[str] = Counter()
    cases = []
    for row in rows:
        if row.get("body_system") != "Integumentary":
            continue
        options = row.get("options")
        if isinstance(options, str):
            options = ast.literal_eval(options)
        if not isinstance(options, dict) or row.get("label") not in options:
            excluded["invalid_options_or_label"] += 1
            continue
        if IMAGE_DEPENDENT.search(row.get("question", "")):
            excluded["image_dependent"] += 1
            continue
        cid = "medxpertqa:" + str(row["id"])
        cases.append({
            "schema_version": VERSION, "case_id": cid, "source_id": cid,
            "source_url": MEDXPERTQA_URL.format(rev=revision), "license": "MIT (dataset card)",
            "source_revision": revision, "source_file_sha256": source_sha256,
            "original_split": "test", "split": "test", "modality": "text",
            "text_origin": "published_exam_question", "gold_origin": "published_answer_key",
            **status, "question": row["question"].strip(),
            "options": {k: str(v).strip() for k, v in options.items()},
            "answer": row["label"],
            "metadata": {"medical_task": row.get("medical_task"),
                         "question_type": row.get("question_type")},
            "limitations": ["Published benchmark; contamination unknown."],
        })
    validate_cases(cases)
    return cases, _report("medxpertqa_derm", cases, excluded,
                          source={"revision": revision, "sha256": source_sha256})


# ════════════════════════════════════════════════════════════════════════
# Track A — SCIN self-reports (closed-set differential)
# ════════════════════════════════════════════════════════════════════════

_DURATION = {
    "ONE_DAY": "about one day", "LESS_THAN_ONE_WEEK": "less than one week",
    "ONE_TO_FOUR_WEEKS": "one to four weeks", "ONE_TO_THREE_MONTHS": "one to three months",
    "THREE_TO_TWELVE_MONTHS": "three to twelve months", "MORE_THAN_ONE_YEAR": "more than one year",
    "MORE_THAN_FIVE_YEARS": "more than five years", "SINCE_CHILDHOOD": "since childhood",
}
_CATEGORY = {
    "RASH": "a rash", "ACNE": "acne or pimples", "GROWTH_OR_MOLE": "a growth or mole",
    "PIGMENTARY_PROBLEM": "a change in skin colour", "NAIL_PROBLEM": "a nail problem",
    "HAIR_LOSS": "hair loss", "LOOKS_HEALTHY": "skin that they think looks healthy",
    "OTHER_ISSUE_DESCRIPTION": "another skin issue",
}


def _yes(row: dict[str, str], prefix: str, skip: tuple[str, ...] = ()) -> list[str]:
    out = []
    for key, val in row.items():
        if key.startswith(prefix) and val == "YES":
            name = key[len(prefix):]
            if name in skip:
                continue
            out.append(name.replace("_or_", " or ").replace("_", " "))
    return out


def scin_case_text(row: dict[str, str]) -> tuple[str, int]:
    """Template a SCIN contributor's structured self-report as plain text.

    Returns the text and the number of informative clinical fields used.
    Only patient-reported fields are used; nothing derived from images or labels.
    """
    who = []
    age = row.get("age_group", "")
    m = re.fullmatch(r"AGE_(\d+)_TO_(\d+)", age)
    if m:
        who.append(f"aged {m.group(1)}-{m.group(2)}")
    elif age == "AGE_80_OR_ABOVE":
        who.append("aged 80 or above")
    sex = row.get("sex_at_birth", "")
    if sex in ("FEMALE", "MALE"):
        who.append(f"{sex.lower()} at birth")
    fst = row.get("fitzpatrick_skin_type", "")
    if re.fullmatch(r"FST[1-6]", fst):
        who.append(f"self-reported Fitzpatrick skin type {'I II III IV V VI'.split()[int(fst[3]) - 1]}")
    lines = ["A person" + (" (" + ", ".join(who) + ")" if who else "") +
             " submitted a skin concern without a clinical examination."]
    informative = 0
    cat = _CATEGORY.get(row.get("related_category", ""))
    if cat:
        lines.append(f"They describe it as {cat}.")
    dur = _DURATION.get(row.get("condition_duration", ""))
    if dur:
        lines.append(f"Duration: {dur}.")
        informative += 1
    for prefix, label, skip in (
        ("body_parts_", "Location", ()),
        ("textures_", "Texture", ()),
        ("condition_symptoms_", "Symptoms at the site", ("no_relevant_experience",)),
        ("other_symptoms_", "Other symptoms", ("no_relevant_symptoms",)),
    ):
        vals = _yes(row, prefix, skip)
        if vals:
            lines.append(f"{label}: {', '.join(vals)}.")
            informative += 1
    if row.get("condition_symptoms_no_relevant_experience") == "YES":
        lines.append("No itching, pain, bleeding, burning, darkening or growth reported.")
    if row.get("other_symptoms_no_relevant_symptoms") == "YES":
        lines.append("No fever, chills, fatigue, joint pain, mouth sores or breathlessness.")
    return " ".join(lines), informative


def _scin_weights(raw: str) -> dict[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    parsed = ast.literal_eval(raw)
    weights = {str(k).strip(): float(v) for k, v in parsed.items() if float(v) > 0}
    total = sum(weights.values())
    # Published weights are rounded (e.g. 0.34+0.33+0.34); renormalize to sum 1.
    return {k: v / total for k, v in weights.items()} if total else {}


def build_scin_text(
    cases_rows: Iterable[dict[str, str]],
    labels_rows: Iterable[dict[str, str]],
    *,
    cases_sha256: str,
    labels_sha256: str,
    top_k: int = 30,
    min_fields: int = 2,
    dev_fraction: float = 0.2,
    salt: str = "dermarbiter-v1",
    data_status: str = "pilot_candidate",
    accepted_by: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Closed-set differential over the ``top_k`` most frequent reference labels.

    Vocabulary = the ``top_k`` labels most often holding the highest dermatologist
    weight (ties count for each tied label). A case is eligible when at least one
    of its highest-weight labels is in the vocabulary and its self-report has at
    least ``min_fields`` informative fields. Non-vocabulary labels keep their
    weight in ``gold_weights`` (unreachable credit, reported honestly).
    """
    status = _status_fields(data_status, accepted_by)
    cases_by_id = {r["case_id"]: r for r in cases_rows}
    excluded: Counter[str] = Counter()
    labeled = []
    for lab in labels_rows:
        weights = _scin_weights(lab.get("weighted_skin_condition_label", ""))
        if not weights:
            excluded["no_dermatologist_label"] += 1
            continue
        row = cases_by_id.get(lab["case_id"])
        if row is None:
            excluded["label_without_case"] += 1
            continue
        labeled.append((row, weights))

    freq: Counter[str] = Counter()
    for _, w in labeled:
        best = max(w.values())
        freq.update(k for k, v in w.items() if abs(v - best) < 1e-9)
    vocab = sorted((k for k, _ in freq.most_common(top_k)), key=str.casefold)
    vocab_set = set(vocab)

    cases = []
    for row, weights in labeled:
        best = max(weights.values())
        tops = {k for k, v in weights.items() if abs(v - best) < 1e-9}
        if not tops & vocab_set:
            excluded["reference_outside_vocabulary"] += 1
            continue
        text, informative = scin_case_text(row)
        if informative < min_fields:
            excluded["too_few_reported_fields"] += 1
            continue
        cid = "scin:" + row["case_id"]
        cases.append({
            "schema_version": DX_VERSION, "case_id": cid, "source_id": cid,
            "source_url": SCIN_URL, "license": "SCIN dataset terms (see source)",
            "source_file_sha256": {"scin_cases.csv": cases_sha256,
                                   "scin_labels.csv": labels_sha256},
            "split": assign_split(norm_text(text), dev_fraction, salt),
            "modality": "text", "text_origin": "patient_reported_structured",
            "gold_origin": "dermatologist_panel_label", **status,
            "case_text": text, "candidates": vocab,
            "gold_weights": weights,
            "metadata": {  # not model-visible; for subgroup analysis only
                "fitzpatrick_self_reported": row.get("fitzpatrick_skin_type") or None,
                "age_group": row.get("age_group") or None,
                "sex_at_birth": row.get("sex_at_birth") or None,
                "combined_race": row.get("combined_race") or None,
                "n_informative_fields": informative,
                "reference_tied": len(tops) > 1,
            },
            "limitations": [
                "Reference labels were assigned by dermatologists from images; "
                "text-only accuracy has a low ceiling by design.",
                "Self-report is templated from structured form fields.",
            ],
        })
    # Identical texts share a split by construction; source ids are unique per case.
    validate_cases(cases)
    report = _report("scin_text", cases, excluded,
                     source={"cases_sha256": cases_sha256, "labels_sha256": labels_sha256})
    report.update(top_k=top_k, min_fields=min_fields, vocabulary=vocab,
                  n_labeled=len(labeled),
                  n_tied_reference=sum(c["metadata"]["reference_tied"] for c in cases))
    return cases, report


def trivial_baselines(cases: list[dict]) -> dict[str, Any]:
    """Accuracy of answer-blind strategies, per split. Report these next to every arm.

    MCQ: always choosing the most frequent answer letter, and uniform guessing.
    DX: always naming the most frequent reference label, and uniform guessing.
    The constant is chosen on dev only so the test figure is not tuned on test.
    """
    from dermarbiter.research.text_benchmark import gold_top, task_of
    dev = [c for c in cases if c["split"] == "dev"] or cases
    first = Counter()
    for c in dev:
        first.update(gold_top(c) if task_of(c) == "dx" else {c["answer"]})
    constant = first.most_common(1)[0][0] if first else None
    out: dict[str, Any] = {"constant_answer_chosen_on_dev": constant}
    for split in ("dev", "test"):
        sub = [c for c in cases if c["split"] == split]
        if not sub:
            continue
        hit = sum(constant in gold_top(c) for c in sub) / len(sub)
        if task_of(sub[0]) == "mcq":
            rand = sum(1 / len(c["options"]) for c in sub) / len(sub)
        else:
            rand = sum(len(gold_top(c) & set(c["candidates"])) / len(c["candidates"])
                       for c in sub) / len(sub)
        out[split] = {"constant_top1": round(hit, 4), "uniform_random_top1": round(rand, 4)}
    return out


def _report(name: str, cases: list[dict], excluded: Counter, source: dict) -> dict[str, Any]:
    splits = Counter(c["split"] for c in cases)
    return {"track": name, "n_cases": len(cases), "splits": dict(splits),
            "excluded": dict(excluded), "source": source,
            "data_status": sorted({c["data_status"] for c in cases}),
            "trivial_baselines": trivial_baselines(cases) if cases else {}}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
