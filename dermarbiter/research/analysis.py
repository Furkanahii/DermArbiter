"""Compare harness arms run on the same cases: accuracy, cost, paired statistics.

Every arm is scored on the same planned cases (missing / error / abstain = not
correct). Differences against a reference arm use a case-level paired bootstrap
and an exact McNemar test, both on the top-1 correctness vectors.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
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
        answers = Counter((r.get("final") or {}).get("answer") for r in recs)
        answers.pop(None, None)
        top_answer, top_n = answers.most_common(1)[0] if answers else (None, 0)
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
            "error_codes": dict(Counter(t["error_code"] for r in recs for t in r.get("trace", [])
                                        if t.get("error_code"))),
            "abstentions": metrics["n_abstain"],
            # Collapse check: share of answered cases that got the single most frequent answer.
            "top_answer": top_answer,
            "top_answer_share": top_n / sum(answers.values()) if answers else None,
            "n_transition_evaluable": metrics["n_transition_evaluable"],
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


REVISING = ("self_refine", "isolated_refine", "debate", "role_debate", "dermarbiter")


def to_markdown(rows: list[dict[str, Any]], reference: str | None = "direct") -> str:
    dx = any("top3" in r for r in rows)
    head = ["model", "arm", "n", "acc"] + (["top3", "soft"] if dx else []) + \
           ["cover", "err", "mode", "rev +/−", "calls/case", "tokens/case", "s/case"]
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
                  pct(r.get("top_answer_share")),
                  f"{r['beneficial']}/{r['harmful']}"
                  if r["arm"] in REVISING and r.get("n_transition_evaluable") else "–",
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
    errs = {f"{r['model']}/{r['arm']}": r["error_codes"] for r in rows if r.get("error_codes")}
    if errs:
        lines.append("\nErrors by arm: " + "; ".join(
            f"{k}: " + ", ".join(f"{c}×{n}" for c, n in v.items()) for k, v in errs.items()))
    if not all(r["publication_eligible"] for r in rows):
        lines.append("\n_Pilot runs on unreviewed dev data: not for publication._")
    if any(not r["tokens_complete"] for r in rows):
        lines.append("_* token usage incomplete for some cases._")
    lines.append("\n_mode: share of answered cases given the most frequent answer. "
                 "rev +/−: initial→final wrong→right / right→wrong (descriptive)._")
    return "\n".join(lines)


def repeat_agreement(cases: list[dict], runs: list[dict]) -> list[dict[str, Any]]:
    """Run-to-run stability: same model and arm, repeat 0 against every later repeat.

    Reports the share of planned cases with the same final answer (abstain/error
    counts as its own value) and the top-1 accuracy difference.
    """
    by_key: dict[tuple, dict[int, dict]] = {}
    for run in runs:
        cfg = run["manifest"]["config"]
        by_key.setdefault((cfg["model"], cfg["method"]), {})[cfg["repeat"]] = run
    ids = [c["case_id"] for c in cases]
    out = []
    for (model, arm), reps in sorted(by_key.items()):
        if 0 not in reps or len(reps) < 2:
            continue
        def finals(run):
            recs = {r["case_id"]: r for r in run["records"]}
            return [(recs[i].get("final") or {}).get("answer") if i in recs else None for i in ids]
        base, base_ok = finals(reps[0]), correctness(cases, reps[0]["records"])
        for rep_i in sorted(k for k in reps if k):
            other, other_ok = finals(reps[rep_i]), correctness(cases, reps[rep_i]["records"])
            out.append({"model": model, "arm": arm, "repeat": rep_i, "n": len(ids),
                        "same_final": sum(a == b for a, b in zip(base, other)) / len(ids),
                        "accuracy_delta": (sum(other_ok) - sum(base_ok)) / len(ids),
                        "correctness_flips": sum(a != b for a, b in zip(base_ok, other_ok))})
    return out


# Primary family (one Holm correction per model): what each step of the
# harness ladder adds over the arm just below it, at matched or explained calls.
PRESPECIFIED = (
    ("isolated_refine", "debate", "peer information at the same topology (6 vs 6 calls)"),
    ("vote6", "debate", "communication vs. sampling at 6 calls"),
    ("vote3", "role_vote", "personas vs. plain samples at 3 calls"),
    ("debate", "role_debate", "personas inside debate at 6 calls"),
    ("role_debate", "dermarbiter", "moderator decision (+1 call)"),
)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def prespecified_comparisons(cases: list[dict], runs: list[dict], n_boot: int = 5000,
                             pairs=PRESPECIFIED) -> list[dict[str, Any]]:
    """Paired top-1 differences for the pre-specified arm pairs, Holm-adjusted per model.

    Uses repeat 0. Pairs with a missing arm are skipped (and so leave the family).
    """
    vec = {}
    for run in runs:
        cfg = run["manifest"]["config"]
        if cfg.get("repeat", 0) == 0:
            vec[(cfg["model"], cfg["method"])] = correctness(cases, run["records"])
    out = []
    for model in sorted({m for m, _ in vec}):
        rows = []
        for base, arm, question in pairs:
            if (model, base) not in vec or (model, arm) not in vec:
                continue
            a, b = vec[(model, base)], vec[(model, arm)]
            m = mcnemar_exact(a, b)
            rows.append({"model": model, "base": base, "arm": arm, "question": question,
                         **paired_bootstrap(a, b, n_boot), "only_base": m["only_reference"],
                         "only_arm": m["only_arm"], "p_value": m["p_value"]})
        for row, adj in zip(rows, holm([r["p_value"] for r in rows])):
            row["p_holm"] = adj
        out.extend(rows)
    return out


def prespecified_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["| model | comparison | Δ top-1 [95% CI] | discordant (base/arm) | p | p (Holm) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['arm']} − {r['base']} | {100 * r['delta']:+.1f} "
                     f"[{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}] | "
                     f"{r['only_base']}/{r['only_arm']} | {r['p_value']:.3f} | {r['p_holm']:.3f} |")
    return "\n".join(lines)


def variance_decomposition(cases: list[dict], runs: list[dict]) -> dict[str, Any]:
    """Share of outcome variance due to case, model, harness and their interactions.

    Balanced design: every (model, harness) cell scored on the same cases, one
    repeat. Outcome = top-1 correctness (0/1). Classical sums of squares for a
    fully crossed case × model × harness layout; the three-way term is the
    residual. ``harness_total`` = harness + model×harness + case×harness, the
    part a harness change can move (cf. MedicalHarness, arXiv:2610.05778).
    Descriptive only: no significance test, and 0/1 outcomes make the shares
    approximate.
    """
    cells: dict[tuple[str, str], list[int]] = {}
    for run in runs:
        cfg = run["manifest"]["config"]
        if cfg.get("repeat", 0) != 0:
            continue
        cells[(cfg["model"], cfg["method"])] = correctness(cases, run["records"])
    models = sorted({m for m, _ in cells})
    arms = sorted({a for _, a in cells})
    if len(models) < 1 or len(arms) < 2:
        raise ValueError("Need at least two harness arms")
    missing = [(m, a) for m in models for a in arms if (m, a) not in cells]
    if missing:
        raise ValueError(f"Unbalanced design, missing cells: {missing[:5]}")
    n_c, n_m, n_a = len(cases), len(models), len(arms)
    y = {(c, m, a): cells[(m, a)][c] for c in range(n_c) for m in models for a in arms}
    grand = sum(y.values()) / len(y)

    def mean(**fix):
        vals = [v for (c, m, a), v in y.items()
                if all({"c": c, "m": m, "a": a}[k] == val for k, val in fix.items())]
        return sum(vals) / len(vals)

    mc = {c: mean(c=c) for c in range(n_c)}
    mm = {m: mean(m=m) for m in models}
    ma = {a: mean(a=a) for a in arms}
    mcm = {(c, m): mean(c=c, m=m) for c in range(n_c) for m in models}
    mca = {(c, a): mean(c=c, a=a) for c in range(n_c) for a in arms}
    mma = {(m, a): mean(m=m, a=a) for m in models for a in arms}
    ss = {
        "case": n_m * n_a * sum((v - grand) ** 2 for v in mc.values()),
        "model": n_c * n_a * sum((v - grand) ** 2 for v in mm.values()),
        "harness": n_c * n_m * sum((v - grand) ** 2 for v in ma.values()),
        "case_x_model": n_a * sum((mcm[c, m] - mc[c] - mm[m] + grand) ** 2
                                  for c in range(n_c) for m in models),
        "case_x_harness": n_m * sum((mca[c, a] - mc[c] - ma[a] + grand) ** 2
                                    for c in range(n_c) for a in arms),
        "model_x_harness": n_c * sum((mma[m, a] - mm[m] - ma[a] + grand) ** 2
                                     for m in models for a in arms),
    }
    total = sum((v - grand) ** 2 for v in y.values())
    ss["residual_3way"] = max(0.0, total - sum(ss.values()))
    share = {k: (v / total if total else 0.0) for k, v in ss.items()}
    share["harness_total"] = share["harness"] + share["model_x_harness"] + share["case_x_harness"]
    return {"n_cases": n_c, "models": models, "harnesses": arms,
            "sum_of_squares": ss, "total_ss": total, "share": share}
