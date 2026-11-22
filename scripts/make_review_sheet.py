"""Draw a fixed random sample of built cases into a CSV for the human data review.

  python scripts/make_review_sheet.py data/text_benchmark/scin_text_k30.jsonl --n 50 \
      --out review/scin_text_k30_review.csv

The sample is a hash order over case ids (same file + same N = same rows), drawn
from all splits. The `shown_to_model` column is exactly what a model sees. The
reviewer fills the last four columns (see docs/data_review_tr.md). Reviewing
does not unlock test runs; freezing does (rebuild with --data-status frozen).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research.text_benchmark import (  # noqa: E402
    digest, gold_top, strict_json, task_of, validate_cases, visible_case,
)

REVIEW_COLUMNS = ("text_ok", "reference_ok", "issue", "comment")


def rows(cases: list[dict], n: int) -> list[dict]:
    sample = sorted(cases, key=lambda c: digest(["review", c["case_id"]]))[:n]
    out = []
    for c in sample:
        shown = visible_case(c)
        if task_of(c) == "mcq":
            text = shown["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(shown["options"].items()))
            ref = f"{c['answer']}. {c['options'][c['answer']]}"
        else:
            text = shown["case"]
            top = gold_top(c)
            ref = "; ".join(f"{k} {v:.2f}" + (" *" if k in top else "")
                            for k, v in sorted(c["gold_weights"].items(), key=lambda kv: -kv[1]))
        out.append({"case_id": c["case_id"], "split": c["split"], "shown_to_model": text,
                    "reference": ref, **{k: "" for k in REVIEW_COLUMNS}})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cases = [strict_json(l) for l in a.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    validate_cases(cases)
    table = rows(cases, a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} rows -> {a.out}")


if __name__ == "__main__":
    main()
