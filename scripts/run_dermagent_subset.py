"""Evaluate DermArbiter on DermAgent's 642-image HAM10000 benchmark subset.

This is the runner that produces the numbers for the **DermArbiter** rows of
``experiments/dermagent_baseline.md`` §5.

Two modes:

    --mock   Use the mock agents + mock tool registry from ``tests/mocks/``.
             Runs in seconds with no API keys or GPU. Useful for shaking out
             the pipeline plumbing before real models are wired in.

    --real   Build the production agents from ``configs/agents.yaml`` via
             the (yet-to-land) factory layer. Requires Furkan's
             ``model_router._call_local`` to be implemented.

The output schema matches what
``dermarbiter.evaluation.metrics.MetricsCalculator.from_jsonl`` expects, so
the metrics can be computed without any additional munging.

Example
-------
::

    # Today — mock pipeline (smoke test, no LLM calls):
    python scripts/run_dermagent_subset.py --mock

    # After real-mode lands:
    python scripts/run_dermagent_subset.py --real --max-cases 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("run_dermagent_subset")

DEFAULT_SUBSET_JSONL = "data/ham10000/dermagent_subset.jsonl"
DEFAULT_OUTPUT_DIR = "experiments/results"

# HAM10000 official 7-class codes ↔ free-text labels the LLM/agent layer
# tends to emit. The benchmark JSONL records ground_truth in HAM10000 codes
# (nv, mel, bkl, …) but agents say "melanocytic_nevus", "seborrheic_keratosis",
# etc. Normalize predictions so apples-to-apples accuracy actually works.
_HAM10000_LABEL_MAP: dict[str, str] = {
    # nv — melanocytic nevus
    "melanocytic_nevus": "nv", "melanocytic nevus": "nv",
    "nevus": "nv", "compound_nevus": "nv", "compound nevus": "nv",
    "atypical_nevus": "nv", "atypical nevus": "nv",
    # mel — melanoma
    "melanoma": "mel", "malignant_melanoma": "mel",
    # bkl — benign keratosis-like (seborrheic, solar lentigo, lichen planus)
    "seborrheic_keratosis": "bkl", "seborrheic keratosis": "bkl",
    "irritated_seborrheic_keratosis": "bkl",
    "irritated seborrheic keratosis": "bkl",
    "inflamed_seborrheic_keratosis": "bkl",
    "inflamed seborrheic keratosis": "bkl",
    "stucco_keratosis": "bkl", "stucco keratosis": "bkl",
    "solar_lentigo": "bkl", "solar lentigo": "bkl",
    "lichen_planus_like_keratosis": "bkl",
    "lichen planus like keratosis": "bkl",
    "lichenoid_keratosis": "bkl", "lichenoid keratosis": "bkl",
    "benign_keratosis": "bkl", "benign keratosis": "bkl",
    "benign_keratosis_like_lesion": "bkl",
    # bcc — basal cell carcinoma
    "basal_cell_carcinoma": "bcc", "basal cell carcinoma": "bcc",
    # akiec — actinic keratosis / Bowen / SCC-in-situ
    "actinic_keratosis": "akiec", "actinic keratosis": "akiec",
    "ak": "akiec",
    "squamous_cell_carcinoma_in_situ": "akiec",
    "squamous cell carcinoma in situ": "akiec",
    "squamous_cell_carcinoma": "akiec",
    "squamous cell carcinoma": "akiec", "scc": "akiec",
    "intraepithelial_carcinoma": "akiec",
    "intraepithelial carcinoma": "akiec",
    "bowen_disease": "akiec", "bowen disease": "akiec", "bowens": "akiec",
    # df — dermatofibroma
    "dermatofibroma": "df",
    # vasc — vascular
    "vascular_lesion": "vasc", "vascular lesion": "vasc",
    "hemangioma": "vasc", "angioma": "vasc",
    # Already-coded inputs pass through.
    "nv": "nv", "mel": "mel", "bkl": "bkl", "bcc": "bcc",
    "akiec": "akiec", "df": "df", "vasc": "vasc",
}


def _normalize_label(raw: str) -> str:
    """Map free-text dermatology labels to HAM10000 7-class codes.

    Case-insensitive, strips punctuation/spaces, returns the raw lowercased
    value when no mapping exists (keeps accuracy honest — unknown labels
    miss instead of silently passing).
    """
    if not raw:
        return ""
    key = raw.strip().lower().replace("-", "_")
    return _HAM10000_LABEL_MAP.get(key, key)


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────


def _to_metrics_record(
    case: dict[str, Any],
    final_diagnosis: list[str],
    consensus_score: float,
    *,
    early_exit: bool,
    debate_rounds: int,
    tool_calls: int,
    total_tokens: int,
    latency_s: float,
    confidence: float | None = None,
    per_class_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Produce one record consumable by MetricsCalculator.from_jsonl()."""
    rec = {
        "case_id": case["case_id"],
        "ground_truth_label": case["ground_truth_label"],
        "predicted_label": final_diagnosis[0] if final_diagnosis else "",
        "top3_predictions": final_diagnosis[:3],
        "consensus_score": consensus_score,
        "early_exit": early_exit,
        "debate_rounds": debate_rounds,
        "tool_calls": tool_calls,
        "total_tokens": total_tokens,
        "latency_s": latency_s,
    }
    if confidence is not None:
        rec["confidence"] = confidence
    if per_class_probs is not None:
        rec["per_class_probs"] = per_class_probs
    if "fitzpatrick_type" in case:
        rec["fitzpatrick_type"] = case["fitzpatrick_type"]
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Mock runner — exercises the runner end-to-end without LLM calls
# ─────────────────────────────────────────────────────────────────────────────


def _run_one_mock(case: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in for a real pipeline run.

    Returns a record with the same shape as a real run would produce. The
    "prediction" is a hash-stable pick from HAM10000 labels — useless as a
    classifier, useful as a plumbing check.
    """
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    # Picking the first class deterministically gives accuracy = 50/642 ≈ 7.8 %
    # for the akiec rows — visible in the metrics summary, which confirms
    # the metrics path is wired correctly.
    pred = classes[hash(case["case_id"]) % len(classes)]

    rec = _to_metrics_record(
        case=case,
        final_diagnosis=[pred, "nv", "bkl"],
        consensus_score=0.6,
        early_exit=True,
        debate_rounds=0,
        tool_calls=3,
        total_tokens=512,
        latency_s=0.001,
        confidence=0.6,
        per_class_probs={c: (1.0 if c == pred else 0.0) for c in classes},
    )

    try:
        from dermarbiter.evaluation.dermabench import state_to_dermabench_prediction
        state = {
            "final_diagnosis": [pred, "nv", "bkl"],
            "consensus_score": 0.6,
            "clinical_report": f"Deterministic mock report. Patient shows {pred} with enlarging and asymmetric features.",
            "briefs": {
                "specialist": type("MockBrief", (), {"cited_cards": ["EC-1", "EC-2"]})(),
            }
        }
        rec["_dermabench"] = state_to_dermabench_prediction(
            case["case_id"], state,
            fitzpatrick_type=case.get("fitzpatrick_type", ""),
            latency_s=0.001,
        )
    except Exception as exc:
        logger.warning("DermAbench mock record build failed for %s: %s",
                       case["case_id"], exc)

    return rec



# ─────────────────────────────────────────────────────────────────────────────
# Real runner — stub until model_router._call_local is implemented
# ─────────────────────────────────────────────────────────────────────────────


def _run_one_real(
    case: dict[str, Any],
    orchestrator: Any,
) -> dict[str, Any]:
    """Run a single case through the real DermArbiter orchestrator.

    The orchestrator is assumed to return a BlackboardState (or dict) with
    fields ``final_diagnosis``, ``consensus_score``, ``early_exit``,
    ``debate_log``, ``total_tool_calls``, ``total_tokens``.
    """
    t0 = time.perf_counter()
    state = orchestrator.invoke({
        "case_id": case["case_id"],
        "image_path": case["image_path"],
        "query": case["query"],
        "patient_context": case.get("patient_context", {}),
    })
    latency_s = time.perf_counter() - t0

    # Coerce dict / Pydantic-model to dict access uniformly.
    get = state.get if isinstance(state, dict) else lambda k, d=None: getattr(state, k, d)

    # Primary path: orchestrator's consensus pick.
    final_dx = list(get("final_diagnosis", []) or [])

    # Fallback 1: empty consensus but moderator's brief has a top-3 → use it.
    # Comes up when 1+ peer agents JSON-parse-fail; the moderator (which still
    # ran successfully because it operates on tool outputs directly) holds the
    # only reliable diagnosis on the blackboard.
    if not final_dx:
        briefs = get("briefs", {}) or {}
        mod_brief = briefs.get("moderator")
        if mod_brief is not None:
            top3 = list(getattr(mod_brief, "top3_differential", []) or [])
            if top3:
                final_dx = top3

    # Normalize labels to HAM10000 codes (agent layer emits free text).
    final_dx_norm = [_normalize_label(d) for d in final_dx if d]

    rec = _to_metrics_record(
        case=case,
        final_diagnosis=final_dx_norm,
        consensus_score=float(get("consensus_score", 0.0) or 0.0),
        early_exit=bool(get("early_exit", False)),
        debate_rounds=len(get("debate_log", []) or []),
        tool_calls=int(get("total_tool_calls", 0) or 0),
        total_tokens=int(get("total_tokens", 0) or 0),
        latency_s=latency_s,
    )

    # Attach a DermAbench prediction record (the right-hand side of the
    # DermAbench scorer join). Carried under a private key so the metrics
    # JSONL stays unchanged; the run loop strips + writes it to a separate
    # file when --dermabench-out is set.
    try:
        from dermarbiter.evaluation.dermabench import state_to_dermabench_prediction
        rec["_dermabench"] = state_to_dermabench_prediction(
            case["case_id"], state,
            fitzpatrick_type=case.get("fitzpatrick_type", ""),
            latency_s=latency_s,
        )
    except Exception as exc:  # never let DermAbench emit break the main run
        logger.warning("DermAbench record build failed for %s: %s",
                       case["case_id"], exc)

    return rec


class _OrchestratorAdapter:
    """Thin wrapper around DermArbiterOrchestrator that accepts a plain dict
    per-case and constructs a fresh ``BlackboardState`` before delegating to
    ``orchestrator.run(state)``. Lets the per-case loop stay agnostic to
    Pydantic types and keeps heavy model state (the agents + tools, with
    cached weights) loaded once across all 642 cases.
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orch = orchestrator

    def invoke(self, case_dict: dict[str, Any]) -> Any:
        from dermarbiter.core.blackboard import BlackboardState
        state = BlackboardState(
            case_id=case_dict["case_id"],
            image_path=case_dict.get("image_path"),
            query=case_dict.get("query", ""),
            patient_context=case_dict.get("patient_context", {}),
        )
        return self._orch.run(state)


def _build_real_orchestrator(args: argparse.Namespace) -> Any:
    """Construct the production orchestrator.

    Mirrors ``scripts/run_e2e_gpu.py::_build_real_pipeline``: loads the
    merged YAML config, instantiates ModelRouter, registers all 9 tools,
    wires up the four agents, and hands them to the LangGraph
    ``DermArbiterOrchestrator``. Heavy models load lazily on first
    ``agent.invoke(...)`` call, then stay cached for every subsequent case
    in the loop — critical when evaluating 642 cases on a T4.
    """
    from dermarbiter.core.config import load_config, AgentConfig
    from dermarbiter.core.model_router import ModelRouter
    from dermarbiter.core.orchestrator import DermArbiterOrchestrator
    from dermarbiter.tools.base_tool import ToolRegistry
    from dermarbiter.agents import (
        SpecialistAgent, GeneralistAgent, SkepticAgent, ModeratorAgent,
    )
    from dermarbiter.tools import (
        PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
        GuidelineRAG, CaseRAG, OntologyGraph, FairnessProbe, UncertaintyProbe,
    )

    config_dir = args.config or "configs/"
    cfg = load_config(config_dir)
    logger.info("Config loaded from %s (project=%s)", config_dir, cfg.project_name)

    router = ModelRouter(cfg)

    # Fill any missing agent config with sensible Gemini defaults so a
    # partial agents.yaml doesn't kill the benchmark.
    agent_configs: dict[str, AgentConfig] = dict(cfg.agents)
    for role in ("specialist", "generalist", "skeptic", "moderator"):
        agent_configs.setdefault(role, AgentConfig(
            role=role,
            model_backend="google_api",
            model_name=cfg.default_model,
            temperature=cfg.default_temperature,
        ))

    # T4 (15 GB) can't hold DermoGPT-RL (~17 GB FP16 / ~7 GB 4-bit) on top of
    # the other tools because the orchestrator doesn't unload between tool
    # calls. Allow opt-out via env var; e.g. `DERMARBITER_DISABLE_TOOLS=dermogpt_vqa`
    # before running this script keeps the rest of the panel intact.
    disabled = {
        t.strip() for t in os.environ.get("DERMARBITER_DISABLE_TOOLS", "").split(",")
        if t.strip()
    }
    if disabled:
        logger.info("Tools disabled via env: %s", sorted(disabled))

    registry = ToolRegistry()
    registered: list[str] = []
    for ToolCls in (PanDermClassifier, MAKEAnnotator, DermoGPTVQA, MedGemmaVQA,
                    GuidelineRAG, CaseRAG, OntologyGraph, FairnessProbe,
                    UncertaintyProbe):
        try:
            inst = ToolCls()
            if inst.name in disabled:
                logger.info("Skipping disabled tool: %s", inst.name)
                continue
            registry.register(inst)
            registered.append(ToolCls.__name__)
        except Exception as exc:
            logger.warning("Skipped tool %s: %s", ToolCls.__name__, exc)
    logger.info("Registered %d tools: %s",
                len(registered), ", ".join(registered))

    agents = {
        "specialist": SpecialistAgent(config=agent_configs["specialist"],
                                       model_router=router,
                                       tool_registry=registry),
        "generalist": GeneralistAgent(config=agent_configs["generalist"],
                                       model_router=router,
                                       tool_registry=registry),
        "skeptic":    SkepticAgent(config=agent_configs["skeptic"],
                                    model_router=router),
        "moderator":  ModeratorAgent(config=agent_configs["moderator"],
                                      model_router=router,
                                      tool_registry=registry),
    }

    orchestrator = DermArbiterOrchestrator(
        agents=agents,
        tool_registry=registry,
        max_rounds=3,
        max_tokens_per_turn=100,
        global_token_budget=50_000,
    )
    return _OrchestratorAdapter(orchestrator)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def _stratified_pick(
    cases: list[dict[str, Any]],
    n: int,
    *,
    key: str = "ground_truth_label",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Class-stratified subsample. Proportional quotas + largest-remainder
    fallback so the per-class quota sums exactly to n. Falls through to a
    plain shuffle if the stratify key is missing.
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    if n >= len(cases):
        return list(cases)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        buckets[(c.get(key) or "unknown")].append(c)

    total = len(cases)
    quotas: dict[str, int] = {}
    remainders: list[tuple[str, float]] = []
    used = 0
    for label, bucket in buckets.items():
        exact = n * len(bucket) / total
        quotas[label] = int(exact)
        remainders.append((label, exact - int(exact)))
        used += quotas[label]
    for label, _ in sorted(remainders, key=lambda x: -x[1]):
        if used >= n:
            break
        if quotas[label] < len(buckets[label]):
            quotas[label] += 1
            used += 1

    sampled: list[dict[str, Any]] = []
    for label, q in quotas.items():
        bucket = buckets[label][:]
        rng.shuffle(bucket)
        sampled.extend(bucket[:q])
    rng.shuffle(sampled)
    return sampled


def load_subset(
    path: Path,
    max_cases: Optional[int] = None,
    *,
    stratified: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load the per-case JSONL.

    Default behaviour: read in file order, stop at ``max_cases``. Use this
    for deterministic small smokes (first N entries are stable).

    ``stratified=True``: read the whole file, then class-stratified-sample
    ``max_cases`` rows so every label is represented proportionally. The
    full HAM10000 dermagent subset is heavily nv-skewed (331/642 are nv)
    and not pre-shuffled, so a naive first-N slice would yield a single-
    class mini-benchmark.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"  python scripts/build_dermagent_subset.py"
        )

    if not stratified:
        cases = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cases.append(json.loads(line))
                if max_cases and len(cases) >= max_cases:
                    break
        return cases

    # Stratified path: full read → proportional sample.
    with path.open("r", encoding="utf-8") as fh:
        all_cases = [json.loads(line) for line in fh if line.strip()]
    if max_cases is None or max_cases >= len(all_cases):
        return all_cases
    return _stratified_pick(all_cases, max_cases, seed=seed)


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Quick in-process metric summary so the runner doesn't need MetricsCalculator."""
    n = len(records)
    if n == 0:
        return {"n_cases": 0}
    correct = sum(1 for r in records if r["predicted_label"] == r["ground_truth_label"])
    top3_correct = sum(
        1 for r in records if r["ground_truth_label"] in (r.get("top3_predictions") or [])
    )
    early_exit = sum(1 for r in records if r.get("early_exit"))
    return {
        "n_cases": n,
        "accuracy": round(correct / n, 4),
        "top3_accuracy": round(top3_correct / n, 4),
        "early_exit_rate": round(early_exit / n, 4),
        "avg_debate_rounds": round(
            sum(r.get("debate_rounds", 0) for r in records) / n, 2
        ),
        "avg_tool_calls": round(
            sum(r.get("tool_calls", 0) for r in records) / n, 2
        ),
        "avg_tokens": round(
            sum(r.get("total_tokens", 0) for r in records) / n, 0
        ),
        "avg_latency_s": round(
            sum(r.get("latency_s", 0.0) for r in records) / n, 3
        ),
    }


def run(args: argparse.Namespace) -> int:
    cases = load_subset(
        Path(args.subset),
        max_cases=args.max_cases,
        stratified=getattr(args, "stratified", False),
        seed=getattr(args, "seed", 42),
    )
    logger.info(
        "Loaded %d cases from %s%s",
        len(cases), args.subset,
        " (stratified)" if getattr(args, "stratified", False) else "",
    )

    # Show per-class distribution of the actual loaded sample so it's
    # obvious whether the mini-benchmark hit every class.
    from collections import Counter
    dist = Counter(c.get("ground_truth_label", "?") for c in cases)
    logger.info("Class distribution: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))

    if args.mock:
        runner = _run_one_mock
        orchestrator = None
    else:
        orchestrator = _build_real_orchestrator(args)
        runner = lambda c: _run_one_real(c, orchestrator)  # noqa: E731

    # Incremental + resumable execution:
    #   * Each completed case is fsync'd to the per-run JSONL immediately
    #     so a Colab disconnect mid-run never loses progress.
    #   * --resume PATH points at an existing JSONL; case_ids already in
    #     that file are skipped and the loop appends new records to the
    #     same file. Without --resume, a fresh timestamped JSONL is
    #     created.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mode = "mock" if args.mock else "real"
    if args.resume:
        jsonl_path = Path(args.resume)
        if not jsonl_path.exists():
            raise SystemExit(f"--resume target not found: {jsonl_path}")
        # Re-derive base + timestamp so the metrics file lands next to it.
        base = jsonl_path.with_suffix("")
        ts = base.name.rsplit("_", 1)[-1]   # "dermagent_subset_real_<ts>"
        completed_records: list[dict[str, Any]] = []
        completed_ids: set[str] = set()
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed_records.append(rec)
            completed_ids.add(rec["case_id"])
        logger.info(
            "Resume: %d cases already done in %s; %d remaining.",
            len(completed_ids), jsonl_path,
            sum(1 for c in cases if c["case_id"] not in completed_ids),
        )
        cases_to_run = [c for c in cases if c["case_id"] not in completed_ids]
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(args.output_dir) / f"dermagent_subset_{mode}_{ts}"
        jsonl_path = base.with_suffix(".jsonl")
        completed_records = []
        cases_to_run = list(cases)
    metrics_path = base.with_suffix(".metrics.json")
    logger.info("Output JSONL: %s  (append mode, fsync per case)", jsonl_path)

    # Optional parallel DermAbench prediction stream.
    dab_path = Path(args.dermabench_out) if getattr(args, "dermabench_out", None) else None
    dab_fh = dab_path.open("a", encoding="utf-8") if dab_path else None
    if dab_fh:
        logger.info("DermAbench predictions → %s", dab_path)

    new_records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        with jsonl_path.open("a", encoding="utf-8") as fh:
            for i, case in enumerate(cases_to_run, start=1):
                try:
                    rec = runner(case)
                except Exception as exc:
                    logger.error("Case %s failed: %s", case["case_id"], exc,
                                 exc_info=args.verbose)
                    rec = {
                        "case_id": case["case_id"],
                        "ground_truth_label": case["ground_truth_label"],
                        "predicted_label": "",
                        "error": str(exc),
                    }
                # Split off the DermAbench record (private key) → its own file.
                dab_rec = rec.pop("_dermabench", None)
                if dab_fh and dab_rec is not None:
                    dab_fh.write(json.dumps(dab_rec, ensure_ascii=False) + "\n")
                    dab_fh.flush()
                    try:
                        os.fsync(dab_fh.fileno())
                    except OSError:
                        pass
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())   # belt + suspenders against Colab disconnect
                except OSError:
                    pass
                new_records.append(rec)
                if i % 10 == 0 or i == len(cases_to_run):
                    logger.info("  progress: %d / %d (new this run)",
                                i, len(cases_to_run))
    finally:
        if dab_fh:
            dab_fh.close()

    elapsed = time.perf_counter() - t0
    records = completed_records + new_records
    logger.info(
        "Run done in %.1fs (%.2f new cases/s) — total in JSONL: %d",
        elapsed,
        len(new_records) / max(elapsed, 1e-9) if new_records else 0,
        len(records),
    )

    summary = {
        "mode": mode,
        "subset": str(args.subset),
        "timestamp_utc": ts,
        "wall_clock_s_this_run": round(elapsed, 2),
        "n_completed_before_this_run": len(completed_records),
        "n_new_this_run": len(new_records),
        **_summarise(records),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f" DermArbiter on DermAgent's 642-image subset — {mode.upper()} mode")
    print("=" * 68)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")
    print(f"\n  Per-case JSONL: {jsonl_path}")
    print(f"  Metrics JSON:   {metrics_path}\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate DermArbiter on DermAgent's 642-image HAM10000 subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subset", default=DEFAULT_SUBSET_JSONL,
                   help="JSONL produced by scripts/build_dermagent_subset.py.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true",
                      help="Use the mock pipeline (no LLM calls).")
    mode.add_argument("--real", action="store_true",
                      help="Use the real orchestrator (needs model_router._call_local).")
    p.add_argument("--config", help="Path to agents.yaml (real mode).")
    p.add_argument("--tools", help="Path to tools.yaml (real mode).")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Cap the number of cases evaluated (smoke test).")
    p.add_argument("--stratified", action="store_true", default=False,
                   help="With --max-cases: class-stratified sample instead of "
                        "the first N entries. Needed because the published "
                        "DermAgent subset isn't pre-shuffled.")
    p.add_argument("--seed", type=int, default=42,
                   help="Stratified sampling seed for reproducibility.")
    p.add_argument("--resume", default=None,
                   help="Path to an existing per-run JSONL. case_ids already "
                        "present are skipped and new records are appended to "
                        "the same file. Use this after a Colab disconnect to "
                        "continue from where you left off without losing work.")
    p.add_argument("--dermabench-out", default=None,
                   help="If set, also write a parallel DermAbench prediction "
                        "JSONL (one record per case) to this path, scoreable "
                        "by dermarbiter.evaluation.dermabench.DermAbenchScorer "
                        "against a frozen gold set. Real-mode only.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
