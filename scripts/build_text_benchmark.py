"""Build the text-only benchmark tracks from local, hash-pinned source files.

Examples (download the sources first; see docs/text_benchmark_runbook.md):

  python scripts/build_text_benchmark.py scin \
      --cases raw/scin/scin_cases.csv --labels raw/scin/scin_labels.csv \
      --output data/text_benchmark/scin_text_k30.jsonl

  python scripts/build_text_benchmark.py medmcqa \
      --parquet raw/medmcqa/train.parquet \
      --revision 91c6572c454088bf71b679ad90aa8dffcd0d5868 \
      --output data/text_benchmark/medmcqa_skin.jsonl

  python scripts/build_text_benchmark.py medxpertqa \
      --jsonl raw/medxpertqa/text_test.jsonl \
      --revision 7e7c465a68eb2b866926bfa59c8c9d17a8daba65 \
      --output data/text_benchmark/medxpertqa_derm.jsonl

Outputs are pilot candidates unless --data-status frozen --accepted-by NAME is
given after the data review. A <output>.report.json provenance file is written
next to every output. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research import datasets as ds  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("track", choices=("scin", "medmcqa", "medxpertqa"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=Path, help="SCIN scin_cases.csv")
    p.add_argument("--labels", type=Path, help="SCIN scin_labels.csv")
    p.add_argument("--parquet", type=Path, help="MedMCQA train parquet")
    p.add_argument("--jsonl", type=Path, help="MedXpertQA Text/test.jsonl")
    p.add_argument("--revision", help="Hugging Face dataset revision (commit sha)")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-fields", type=int, default=2)
    p.add_argument("--dev-fraction", type=float, default=0.2)
    p.add_argument("--salt", default="dermarbiter-v1")
    p.add_argument("--data-status", choices=("pilot_candidate", "frozen"), default="pilot_candidate")
    p.add_argument("--accepted-by")
    p.add_argument("--keep-option-order", action="store_true",
                   help="MedMCQA: keep the published A-D order (default shuffles per question)")
    a = p.parse_args(argv)
    status = {"data_status": a.data_status, "accepted_by": a.accepted_by}

    if a.track == "scin":
        if not (a.cases and a.labels):
            p.error("scin requires --cases and --labels")
        cases, report = ds.build_scin_text(
            ds.read_csv(a.cases), ds.read_csv(a.labels),
            cases_sha256=ds.sha256_file(a.cases), labels_sha256=ds.sha256_file(a.labels),
            top_k=a.top_k, min_fields=a.min_fields, dev_fraction=a.dev_fraction,
            salt=a.salt, **status)
    elif a.track == "medmcqa":
        if not (a.parquet and a.revision):
            p.error("medmcqa requires --parquet and --revision")
        import pyarrow.parquet as pq
        cases, report = ds.build_medmcqa_skin(
            pq.read_table(a.parquet).to_pylist(), revision=a.revision,
            source_sha256=ds.sha256_file(a.parquet), dev_fraction=a.dev_fraction,
            salt=a.salt, shuffle_options=not a.keep_option_order, **status)
    else:
        if not (a.jsonl and a.revision):
            p.error("medxpertqa requires --jsonl and --revision")
        rows = [json.loads(line) for line in a.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        cases, report = ds.build_medxpertqa_derm(
            rows, revision=a.revision, source_sha256=ds.sha256_file(a.jsonl), **status)

    ds.write_jsonl(a.output, cases)
    report["output"] = str(a.output)
    report["output_sha256"] = ds.sha256_file(a.output)
    a.output.with_suffix(a.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("track", "n_cases", "splits", "excluded",
                                    "trivial_baselines", "output_sha256")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
