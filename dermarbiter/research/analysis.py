"""Compare harness arms run on the same cases: accuracy, cost, paired statistics.

Every arm is scored on the same planned cases (missing / error / abstain = not
correct). Differences against a reference arm use a case-level paired bootstrap
and an exact McNemar test, both on the top-1 correctness vectors.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from dermarbiter.research.text_benchmark import evaluate, gold_top, strict_json, task_of


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {"dir": str(run_dir), "manifest": manifest,
            "records": read_jsonl(run_dir / "predictions.jsonl")}


def correctness(cases: list[dict], records: list[dict]) -> list[int]:
    """Top-1 correctness per planned case, in case order. Missing counts as 0."""
    by_id = {r["case_id"]: r for r in records}
    out = []
    for case in cases:
        r = by_id.get(case["case_id"])
        answer = (r.get("final") or {}).get("answer") if r and r.get("status") == "ok" else None
        out.append(int(answer is not None and answer in gold_top(case)))
    return out


def paired_bootstrap(a: list[int], b: list[int], n_boot: int = 5000,
                     seed: int = 0) -> dict[str, float]:
    """Mean of (b - a) with a percentile 95% CI, resampling cases with replacement."""
    if len(a) != len(b) or not a:
        raise ValueError("Paired vectors must be non-empty and the same length")
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"delta": sum(diffs) / n,
            "ci_low": means[int(0.025 * n_boot)], "ci_high": means[int(0.975 * n_boot) - 1]}


def mcnemar_exact(a: list[int], b: list[int]) -> dict[str, float]:
    """Exact two-sided McNemar test on discordant pairs."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_reference": 0, "only_arm": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_reference": only_a, "only_arm": only_b, "p_value": min(1.0, 2 * tail)}


def summarize(cases: list[dict], runs: list[dict], reference: str | None = "direct",
              n_boot: int = 5000) -> list[dict[str, Any]]:
    """One row per run. Rows sharing model+repeat with the reference arm get deltas."""
    task = task_of(cases[0])
    rows = []
    vectors = {}
    for run in runs:
        cfg = run["manifest"]["config"]
        metrics = evaluate(cases, run["records"])
        recs = run["records"]
        tokens = [r["total_tokens"] for r in recs if r.get("total_tokens") is not None]
        row = {
            "arm": cfg["method"], "model": cfg["model"], "repeat": cfg["repeat"],
            "n": metrics["n_planned"], "accuracy": metrics["accuracy_all_cases"],
            "coverage": metrics["answer_coverage"], "errors": metrics["n_error"],
            "missing": metrics["n_missing"],
            "beneficial": metrics["beneficial_revisions"], "harmful": metrics["harmful_revisions"],
            "calls_per_case": (sum(r["calls_attempted"] for r in recs) / len(recs)) if recs else None,
            "tokens_per_case": (sum(tokens) / len(tokens)) if tokens else None,
            "tokens_complete": len(tokens) == len(recs) and bool(recs),
            "latency_per_case_s": (sum(r["latency_seconds"] for r in recs) / len(recs)) if recs else None,
            "publication_eligible": bool(run["manifest"].get("publication_eligible", False)),
            "run_dir": run["dir"],
        }
        if task == "dx":
            row["top3"] = metrics["top3_accuracy_all_cases"]
            row["top1_soft"] = metrics["top1_soft_all_cases"]
        vectors[(cfg["model"], cfg["repeat"], cfg["method"])] = correctness(cases, recs)
        rows.append(row)
    if reference:
        for row in rows:
            ref = vectors.get((row["model"], row["repeat"], reference))
            if ref is None or row["arm"] == reference:
                continue
            arm = vectors[(row["model"], row["repeat"], row["arm"])]
            row.update({f"vs_{reference}_" + k: v for k, v in paired_bootstrap(ref, arm, n_boot).items()})
            row[f"vs_{reference}_p"] = mcnemar_exact(ref, arm)["p_value"]
    order = {m: i for i, m in enumerate(
        ["direct", "structured", "self_refine", "vote3", "vote6", "isolated_refine", "debate",
         "role_vote", "role_debate", "dermarbiter"])}
    rows.sort(key=lambda r: (r["model"], r["repeat"], order.get(r["arm"], 99)))
    return rows


def to_markdown(rows: list[dict[str, Any]], reference: str | None = "direct") -> str:
    dx = any("top3" in r for r in rows)
    head = ["model", "arm", "n", "acc"] + (["top3", "soft"] if dx else []) + \
           ["cover", "err", "calls/case", "tokens/case", "s/case"]
    if reference:
        head += [f"Δ vs {reference} [95% CI]", "McNemar p"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}"
    for r in rows:
        cells = [r["model"] + (f" r{r['repeat']}" if r["repeat"] else ""), r["arm"], str(r["n"]),
                 pct(r["accuracy"])]
        if dx:
            cells += [pct(r.get("top3")), pct(r.get("top1_soft"))]
        cells += [pct(r["coverage"]), str(r["errors"] + r["missing"]),
                  "–" if r["calls_per_case"] is None else f"{r['calls_per_case']:.1f}",
                  "–" if r["tokens_per_case"] is None else f"{r['tokens_per_case']:.0f}"
                  + ("" if r["tokens_complete"] else "*"),
                  "–" if r["latency_per_case_s"] is None else f"{r['latency_per_case_s']:.1f}"]
        if reference:
            key = f"vs_{reference}_delta"
            if key in r:
                cells += [f"{100 * r[key]:+.1f} [{100 * r[f'vs_{reference}_ci_low']:+.1f}, "
                          f"{100 * r[f'vs_{reference}_ci_high']:+.1f}]",
                          f"{r[f'vs_{reference}_p']:.3f}"]
            else:
                cells += ["ref" if r["arm"] == reference else "–", ""]
        lines.append("| " + " | ".join(cells) + " |")
    if not all(r["publication_eligible"] for r in rows):
        lines.append("\n_Pilot runs on unreviewed dev data: not for publication._")
    if any(not r["tokens_complete"] for r in rows):
        lines.append("_* token usage incomplete for some cases._")
    return "\n".join(lines)
