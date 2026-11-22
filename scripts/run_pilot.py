"""Run several harness arms on the same cases, then print the comparison table.

Each arm is a separate run_text_benchmark.py run under OUT/<model>/<arm>_r<repeat>.
Arms whose run already completed are skipped, so an interrupted pilot can be
continued by re-running the same command. Unfinished run folders stop the
pilot: audit and delete them by hand (no silent overwrite).

Example (Gemini, 20 SCIN dev cases, all arms):

  export GEMINI_API_KEY=...        # in your own terminal, never in a file in the repo
  python scripts/run_pilot.py data/text_benchmark/scin_text_k30.jsonl \
      --split dev --limit 20 --dev-pilot \
      --model gemini-2.5-flash --expected-model-version gemini-2.5-flash \
      --out runs/pilot_scin_dev20

Use --plan-only first to see the call count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dermarbiter.research.analysis import load_run, read_jsonl, summarize, to_markdown  # noqa: E402
from dermarbiter.research.text_benchmark import METHOD_CALLS, digest, validate_cases  # noqa: E402

RUNNER = ROOT / "scripts" / "run_text_benchmark.py"


def select(cases, split, limit):
    sel = [c for c in cases if c["split"] == split]
    if limit:
        sel = sorted(sel, key=lambda c: digest(["pilot-order", c["case_id"]]))[:limit]
    return sel


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--arms", default=",".join(METHOD_CALLS),
                   help="Comma-separated arms (default: all)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", required=True)
    p.add_argument("--expected-model-version", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", default="direct")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--dev-pilot", action="store_true")
    p.add_argument("--provider", choices=("gemini", "openai_compatible"), default="gemini")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--token-param", choices=("max_tokens", "max_completion_tokens"))
    p.add_argument("--no-json-mode", action="store_true")
    a = p.parse_args(argv)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = set(arms) - set(METHOD_CALLS)
    if unknown:
        p.error(f"Unknown arms: {sorted(unknown)}")
    cases = read_jsonl(a.dataset)
    validate_cases(cases)
    selected = select(cases, a.split, a.limit)
    total = len(selected) * sum(METHOD_CALLS[m] for m in arms) * a.repeats
    print(json.dumps({"n_cases": len(selected), "arms": arms, "repeats": a.repeats,
                      "maximum_calls": total}, indent=2))
    if a.plan_only:
        return

    model_dir = a.out / a.model.replace("/", "__")
    runs = []
    for repeat in range(a.repeats):
        for arm in arms:
            run_dir = model_dir / f"{arm}_r{repeat}"
            manifest = run_dir / "manifest.json"
            if run_dir.exists():
                status = json.loads(manifest.read_text()).get("status") if manifest.exists() else None
                if status != "completed":
                    sys.exit(f"{run_dir} exists but did not complete; audit and remove it first")
                print(f"skip {arm} r{repeat} (completed)")
                runs.append(run_dir)
                continue
            cmd = [sys.executable, str(RUNNER), "run", str(a.dataset), "--split", a.split,
                   "--method", arm, "--model", a.model,
                   "--expected-model-version", a.expected_model_version,
                   "--repeat", str(repeat), "--temperature", str(a.temperature),
                   "--max-output-tokens", str(a.max_output_tokens),
                   "--max-calls", str(len(selected) * METHOD_CALLS[arm]),
                   "--output-dir", str(run_dir), "--provider", a.provider]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            if a.dev_pilot:
                cmd.append("--dev-pilot")
            for flag, val in (("--base-url", a.base_url), ("--api-key-env", a.api_key_env),
                              ("--token-param", a.token_param)):
                if val:
                    cmd += [flag, val]
            if a.no_json_mode:
                cmd.append("--no-json-mode")
            print(f"run {arm} r{repeat} ...", flush=True)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            runs.append(run_dir)

    rows = summarize(selected, [load_run(r) for r in runs], reference=a.reference)
    table = to_markdown(rows, reference=a.reference)
    (model_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (model_dir / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
