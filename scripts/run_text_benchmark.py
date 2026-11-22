"""Plan, run or score a text-only harness arm (MCQ or closed-set dx). Planning never calls a model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research.text_benchmark import (  # noqa: E402
    Config, GeminiProvider, METHOD_CALLS, OpenAICompatibleProvider, digest, evaluate,
    run_case, strict_json, task_of, validate_cases,
)


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "score"))
    p.add_argument("dataset", type=Path)
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--model")
    p.add_argument("--expected-model-version")
    p.add_argument("--method", choices=METHOD_CALLS, default="direct")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-output-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--max-calls", type=int)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--provider", choices=("gemini", "openai_compatible"), default="gemini")
    p.add_argument("--base-url", help="openai_compatible: e.g. https://api.openai.com/v1 "
                   "or http://localhost:8000/v1 for vLLM")
    p.add_argument("--api-key-env", help="Name of the env var holding the API key "
                   "(default GEMINI_API_KEY / OPENAI_API_KEY). Keys are never passed as arguments.")
    p.add_argument("--token-param", choices=("max_tokens", "max_completion_tokens"),
                   default="max_tokens")
    p.add_argument("--no-json-mode", action="store_true",
                   help="openai_compatible: do not send response_format=json_object")
    args = p.parse_args()
    cases = read_jsonl(args.dataset)
    validate_cases(cases)  # cross-split checks precede selection
    selected = [c for c in cases if c["split"] == args.split]
    if not selected:
        p.error("Selected split is empty")
    if args.action == "score":
        if not args.predictions:
            p.error("score requires --predictions")
        predictions = read_jsonl(args.predictions)
        dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if any(r.get("dataset_sha256") != dataset_hash for r in predictions):
            p.error("Prediction dataset hash mismatch: use the exact frozen input file")
        print(json.dumps(evaluate(selected, predictions), indent=2))
        return
    calls = len(selected) * METHOD_CALLS[args.method]
    plan = {"schema_version": selected[0]["schema_version"], "task": task_of(selected[0]),
            "method": args.method, "split": args.split,
            "n_cases": len(selected), "maximum_calls": calls,
            "nominal_output_token_allowance": calls * args.max_output_tokens,
            "budget_note": "Not a dollar cap or matched total-token budget; input/thinking usage varies.",
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "selected_case_ids": [c["case_id"] for c in selected]}
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return
    if not args.output_dir or not args.model or not args.expected_model_version:
        p.error("run requires --output-dir, --model and --expected-model-version")
    if args.max_calls is None or calls > args.max_calls:
        p.error(f"run requires an explicit --max-calls >= {calls}")
    if any(c["text_origin"] == "fixture" or c["gold_origin"] == "fixture" for c in selected):
        p.error("Fixtures are for local tests, not live research runs")
    if any(c["data_status"] != "frozen" for c in selected):
        p.error("Data acceptance review is pending: live runs require data_status=frozen")
    config = Config(args.model, args.expected_model_version, args.method,
                    args.temperature, args.max_output_tokens, args.seed, args.repeat)
    if args.provider == "gemini":
        provider = GeminiProvider(os.environ.get(args.api_key_env or "GEMINI_API_KEY", ""))
    else:
        if not args.base_url:
            p.error("openai_compatible requires --base-url")
        provider = OpenAICompatibleProvider(
            args.base_url, os.environ.get(args.api_key_env or "OPENAI_API_KEY", ""),
            json_mode=not args.no_json_mode, token_param=args.token_param)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).resolve().parents[1] / "dermarbiter/research/text_benchmark.py"
    manifest = {**plan, "config": asdict(config), "config_sha256": digest(asdict(config)),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "code_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "started", "provider": provider.name,
                "base_url": getattr(provider, "base_url", None), "retries": 0,
                "publication_review_required": True}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as out:
        for case in selected:
            record = run_case(case, config, provider)
            record["dataset_sha256"] = plan["dataset_sha256"]
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            records.append(record)
            print(f"{len(records)}/{len(selected)}: {record['status']}", flush=True)
    metrics = evaluate(selected, records)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
