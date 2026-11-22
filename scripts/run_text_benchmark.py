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
    TASK_SYSTEM, Config, GeminiProvider, METHOD_CALLS, OpenAICompatibleProvider, digest,
    evaluate, response_schema, run_case, strict_json, task_of, validate_cases,
    visible_case,
)


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _provider(args, p):
    if args.provider == "gemini":
        return GeminiProvider(os.environ.get(args.api_key_env or "GEMINI_API_KEY", ""),
                              timeout=args.timeout)
    if not args.base_url:
        p.error("openai_compatible requires --base-url")
    return OpenAICompatibleProvider(
        args.base_url, os.environ.get(args.api_key_env or "OPENAI_API_KEY", ""),
        timeout=args.timeout, json_mode=not args.no_json_mode, token_param=args.token_param)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "probe", "run", "score"))
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
    p.add_argument("--limit", type=int,
                   help="Use only the first N cases of the split in a fixed hash order "
                        "(pilots). The same N always selects the same cases.")
    p.add_argument("--dev-pilot", action="store_true",
                   help="Allow pilot_candidate data on the dev split only. Runs are "
                        "marked publication_eligible=false.")
    p.add_argument("--timeout", type=float, default=120,
                   help="Per-call HTTP timeout in seconds (raise it for slow local hardware)")
    p.add_argument("--no-json-mode", action="store_true",
                   help="openai_compatible: do not send response_format=json_object")
    args = p.parse_args(argv)
    cases = read_jsonl(args.dataset)
    validate_cases(cases)  # cross-split checks precede selection
    selected = [c for c in cases if c["split"] == args.split]
    if args.limit is not None:
        if args.limit < 1:
            p.error("--limit must be positive")
        selected = sorted(selected, key=lambda c: digest(["pilot-order", c["case_id"]]))[:args.limit]
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
    if args.action == "probe":
        # One live call on the first selected case; prints the returned model version
        # to pin with --expected-model-version. Writes nothing.
        if not args.model:
            p.error("probe requires --model")
        provider = _provider(args, p)
        system = TASK_SYSTEM[task_of(selected[0])]
        prompt = "Rank the most likely conditions." if task_of(selected[0]) == "dx" else "Select the best answer."
        prompt += "\nCASE:\n" + json.dumps(visible_case(selected[0]), ensure_ascii=False)
        probe_cfg = Config(args.model, "probe", "direct", args.temperature, args.max_output_tokens)
        schema = response_schema(selected[0]) if getattr(provider, "accepts_schema", False) else None
        c = (provider.complete(system, prompt, probe_cfg, 0, schema=schema) if schema
             else provider.complete(system, prompt, probe_cfg, 0))
        print(json.dumps({"returned_model_version": c.model_version, "finish_reason": c.finish_reason,
                          "usage": c.usage, "text_preview": c.text[:300]}, indent=2))
        return
    if not args.output_dir or not args.model or not args.expected_model_version:
        p.error("run requires --output-dir, --model and --expected-model-version")
    if args.max_calls is None or calls > args.max_calls:
        p.error(f"run requires an explicit --max-calls >= {calls}")
    if any(c["text_origin"] == "fixture" or c["gold_origin"] == "fixture" for c in selected):
        p.error("Fixtures are for local tests, not live research runs")
    unfrozen = any(c["data_status"] != "frozen" for c in selected)
    if unfrozen and not (args.dev_pilot and args.split == "dev"):
        p.error("Data acceptance review is pending: live runs require data_status=frozen "
                "(or --dev-pilot on the dev split)")
    config = Config(args.model, args.expected_model_version, args.method,
                    args.temperature, args.max_output_tokens, args.seed, args.repeat)
    provider = _provider(args, p)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    module_path = Path(__file__).resolve().parents[1] / "dermarbiter/research/text_benchmark.py"
    manifest = {**plan, "config": asdict(config), "config_sha256": digest(asdict(config)),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "code_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "started", "provider": provider.name,
                "base_url": getattr(provider, "base_url", None), "retries": 0,
                "timeout_seconds": args.timeout,
                "publication_review_required": True,
                "limit": args.limit, "dev_pilot": bool(unfrozen),
                "publication_eligible": not unfrozen}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as out:
        for case in selected:
            record = run_case(case, config, provider)
            record["dataset_sha256"] = plan["dataset_sha256"]
            record["publication_eligible"] = not unfrozen
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
