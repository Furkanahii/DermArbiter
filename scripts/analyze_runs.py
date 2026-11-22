"""Cross-model comparison table + variance decomposition for completed runs.

  python scripts/analyze_runs.py data/text_benchmark/scin_text_k30.jsonl --split dev --limit 10 \
      --runs runs/pilot/scin_dev10/*/* --out runs/pilot/scin_dev10/analysis

Every run directory must hold manifest.json + predictions.jsonl for the same
dataset, split and --limit. Writes <out>.md and <out>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research.analysis import (  # noqa: E402
    load_run, prespecified_comparisons, prespecified_markdown, read_jsonl, summarize,
    to_markdown, variance_decomposition,
)
from dermarbiter.research.text_benchmark import digest, validate_cases  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--reference", default="direct")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cases = read_jsonl(a.dataset)
    validate_cases(cases)
    sel = [c for c in cases if c["split"] == a.split]
    if a.limit:
        sel = sorted(sel, key=lambda c: digest(["pilot-order", c["case_id"]]))[:a.limit]
    runs = [load_run(r) for r in a.runs if (r / "manifest.json").exists()]
    runs = [r for r in runs if r["manifest"].get("status") == "completed"]
    rows = summarize(sel, runs, reference=a.reference)
    text = [to_markdown(rows, reference=a.reference)]
    out = {"rows": rows}
    pre = prespecified_comparisons(sel, runs)
    if pre:
        out["prespecified"] = pre
        text.append("\n**Pre-specified comparisons** (Holm within each model)\n")
        text.append(prespecified_markdown(pre))
    try:
        vd = variance_decomposition(sel, runs)
        out["variance_decomposition"] = vd
        sh = vd["share"]
        text.append("\n**Variance decomposition of top-1 correctness** "
                    f"({vd['n_cases']} cases × {len(vd['models'])} models × {len(vd['harnesses'])} harnesses)\n")
        text.append("| component | share |\n|---|---|")
        for k in ("case", "model", "harness", "model_x_harness", "case_x_model",
                  "case_x_harness", "residual_3way", "harness_total"):
            text.append(f"| {k} | {100 * sh[k]:.1f}% |")
    except ValueError as exc:
        text.append(f"\n_Variance decomposition skipped: {exc}_")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".md").write_text("\n".join(text) + "\n")
    Path(str(a.out) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
