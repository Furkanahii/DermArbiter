#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
DermArbiter Full Pipeline Demo
═══════════════════════════════════════════════════════════════
This demo runs the complete DermArbiter multi-agent diagnostic
pipeline using mock agents and tools — no GPU or API keys.

Sections:
  1. Single Case Demo          — Full 5-phase pipeline on one case
  2. Batch Benchmark           — All 5 sample cases via ExperimentRunner
  3. Metrics Analysis          — Accuracy, F1, ECE, Brier, confusion matrix
  4. Fairness Analysis         — Per-group metrics across Fitzpatrick types
  5. Ablation Preview          — Agent/tool/round ablation configuration
  6. Architecture Visualization — ASCII pipeline diagram

For GPU execution, use:  python scripts/run_e2e_gpu.py --mock

Usage:
    python notebooks/03_full_pipeline_demo.py
═══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress noisy debug logs — keep only warnings+
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _banner(title: str, char: str = "═", width: int = 68) -> None:
    """Print a section banner."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def _sub_header(title: str, char: str = "─", width: int = 60) -> None:
    """Print a sub-section header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}\n")


def _table_row(*cols: str, widths: list[int] | None = None) -> str:
    """Format a row with fixed-width columns."""
    if widths is None:
        return "  " + "  ".join(cols)
    parts = []
    for col, w in zip(cols, widths):
        parts.append(f"{col:<{w}s}")
    return "  " + "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Single Case Demo
# ═══════════════════════════════════════════════════════════════════════════

def section_1_single_case() -> Any:
    """
    Run a single case through the complete 5-phase DermArbiter pipeline:
      Phase 1: Plan & Probe     — tool proposals + batch execution
      Phase 2: Independent Read — agents generate diagnostic briefs
      Phase 3: Reveal & Critique — early exit gating
      Phase 4: Targeted Debate  — structured argument rounds
      Phase 5: Synthesis        — consensus ranking + clinical report
    """
    _banner("SECTION 1: Single Case — Full 5-Phase Pipeline")

    from dermarbiter.core.blackboard import BlackboardState
    from dermarbiter.core.debate_protocol import (
        plan_probe,
        independent_read,
        reveal_critique,
        targeted_debate,
        synthesis,
    )
    from tests.mocks.mock_agents import create_mock_agents
    from tests.mocks.mock_tools import create_mock_registry

    # ── Setup ──
    agents = create_mock_agents()
    registry = create_mock_registry()

    state = BlackboardState(
        case_id="DEMO-001",
        query="55yo male, changing pigmented lesion upper back, 6 months",
        image_path=None,
        patient_context={
            "age": 55,
            "sex": "Male",
            "fitzpatrick_type": "III",
            "location": "upper back",
        },
    )

    print(f"  Case ID:  {state.case_id}")
    print(f"  Query:    {state.query}")
    print(f"  Context:  {state.patient_context}")
    print(f"  Agents:   {list(agents.keys())}")
    print(f"  Tools:    {len(registry)} registered")

    t_total = time.time()

    # ── Phase 1: Plan & Probe ──
    _sub_header("Phase 1: Plan & Probe")
    t0 = time.time()
    plan_probe(state, agents, registry)
    t1 = time.time()
    print(f"  Evidence cards collected: {len(state.evidence_cards)}")
    print(f"  Time: {(t1 - t0) * 1000:.1f}ms\n")

    for card in state.evidence_cards:
        to = card.tool_output
        print(f"    [{card.card_id}] {to.tool_name:<22s} "
              f"conf={to.confidence:.2f}  by={card.requested_by}")

    # ── Phase 2: Independent Reading ──
    _sub_header("Phase 2: Independent Reading")
    t0 = time.time()
    independent_read(state, agents)
    t1 = time.time()
    print(f"  Briefs submitted: {list(state.briefs.keys())}")
    print(f"  Time: {(t1 - t0) * 1000:.1f}ms\n")

    for role, brief in state.briefs.items():
        dx = ", ".join(brief.top3_differential)
        flags = ", ".join(brief.disagreement_flags) or "(none)"
        print(f"    [{role.upper():>12s}]  Dx: {dx}")
        print(f"    {'':>14s}  Confidence: {brief.confidence:.2f}  |  Flags: {flags}")

    # ── Phase 3: Reveal & Critique ──
    _sub_header("Phase 3: Reveal & Critique (Early Exit Gate)")
    t0 = time.time()
    reveal_critique(state, agents)
    t1 = time.time()
    print(f"  Early exit triggered: {state.early_exit}")
    print(f"  Time: {(t1 - t0) * 1000:.1f}ms")

    # ── Phase 4: Targeted Debate ──
    _sub_header("Phase 4: Targeted Debate")
    t0 = time.time()
    if state.early_exit:
        print("  ⚡ Skipped — consensus reached in Phase 3")
    else:
        targeted_debate(state, agents, max_rounds=3)
    t1 = time.time()
    print(f"  Debate turns: {len(state.debate_log)}")
    print(f"  Time: {(t1 - t0) * 1000:.1f}ms\n")

    for turn in state.debate_log:
        arg_preview = turn.argument[:100] + "…" if len(turn.argument) > 100 else turn.argument
        print(f"    R{turn.round_num} [{turn.speaker:>12s}] "
              f"({turn.token_count} tok): {arg_preview}")

    # ── Phase 5: Synthesis ──
    _sub_header("Phase 5: Synthesis")
    t0 = time.time()
    synthesis(state, agents)
    t1 = time.time()

    print(f"  Final diagnoses (ranked):")
    for i, dx in enumerate(state.final_diagnosis, 1):
        print(f"    {i}. {dx.title()}")

    print(f"\n  Consensus score: {state.consensus_score:.2f}")
    print(f"  Dissent notes:   {state.dissent_notes or '(none)'}")
    print(f"  Time: {(t1 - t0) * 1000:.1f}ms")

    # ── Clinical Report ──
    _sub_header("Clinical Report")
    if state.clinical_report:
        for line in state.clinical_report.strip().split("\n"):
            print(f"  {line}")
    else:
        print("  (no report generated)")

    # ── Blackboard State Summary ──
    _sub_header("Full Blackboard State")
    elapsed = time.time() - t_total
    print(f"  case_id:           {state.case_id}")
    print(f"  evidence_cards:    {len(state.evidence_cards)}")
    print(f"  briefs:            {list(state.briefs.keys())}")
    print(f"  debate_log:        {len(state.debate_log)} turns")
    print(f"  early_exit:        {state.early_exit}")
    print(f"  final_diagnosis:   {state.final_diagnosis}")
    print(f"  consensus_score:   {state.consensus_score:.2f}")
    print(f"  dissent_notes:     {len(state.dissent_notes)}")
    print(f"  total_tokens:      {state.total_tokens}")
    print(f"  total_tool_calls:  {state.total_tool_calls}")
    print(f"  errors:            {state.errors}")
    print(f"  total_time:        {elapsed:.2f}s")

    return state


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Batch Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def section_2_batch_benchmark() -> List[Dict[str, Any]]:
    """
    Run all 5 sample cases from data/sample_cases.jsonl through the
    ExperimentRunner in mock mode and display per-case results.
    """
    _banner("SECTION 2: Batch Benchmark (5 Sample Cases)")

    from dermarbiter.experiments.runner import ExperimentRunner

    data_path = str(_PROJECT_ROOT / "data" / "sample_cases.jsonl")
    output_path = str(_PROJECT_ROOT / "results" / "demo_benchmark.jsonl")

    print(f"  Dataset:  {data_path}")
    print(f"  Output:   {output_path}")
    print(f"  Mode:     Mock (CPU-only)\n")

    t0 = time.time()
    runner = ExperimentRunner(
        config_path="configs/",
        data_path=data_path,
        output_path=output_path,
        mock=True,
    )
    results = runner.run()
    t1 = time.time()

    # Print results table
    widths = [12, 28, 28, 6, 6, 7, 5]
    header = _table_row("Case ID", "Predicted", "Ground Truth",
                        "Cons.", "Exit?", "Rounds", "Tok",
                        widths=widths)
    sep = _table_row(*["─" * w for w in widths], widths=widths)
    print(header)
    print(sep)

    for r in results:
        predicted = r.get("predicted", "")[:28]
        gt = r.get("ground_truth", "")[:28]
        correct = "✓" if predicted.lower() == gt.lower() else "✗"
        print(_table_row(
            r.get("case_id", ""),
            f"{predicted} {correct}",
            gt,
            f"{r.get('consensus_score', 0):.2f}",
            "Y" if r.get("early_exit") else "N",
            str(r.get("num_debate_rounds", 0)),
            str(r.get("total_tokens", 0)),
            widths=widths,
        ))

    correct_count = sum(
        1 for r in results
        if r.get("predicted", "").strip().lower() == r.get("ground_truth", "").strip().lower()
    )
    print(f"\n  Accuracy: {correct_count}/{len(results)} ({correct_count/max(len(results),1):.0%})")
    print(f"  Total time: {(t1 - t0):.2f}s")
    print(f"  Avg latency: {sum(r.get('latency_ms', 0) for r in results)/max(len(results),1):.1f}ms")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Metrics Analysis
# ═══════════════════════════════════════════════════════════════════════════

def section_3_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Compute comprehensive evaluation metrics using both:
      • ResultsAnalyzer (pure-Python, from experiments module)
      • MetricsCalculator (numpy-backed, from evaluation module)
    """
    _banner("SECTION 3: Metrics Analysis")

    # ── Part A: ResultsAnalyzer ──
    _sub_header("Part A: ResultsAnalyzer (Pure Python)")

    from dermarbiter.experiments.analyze import ResultsAnalyzer

    analyzer = ResultsAnalyzer()
    analyzer.load_records(results)

    metrics = analyzer.to_dict()
    print(f"  Cases evaluated:     {metrics['n_cases']}")
    print(f"  Top-1 Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy:      {metrics['top3_accuracy']:.4f}")
    print(f"  Macro-F1:            {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:         {metrics['weighted_f1']:.4f}")
    print(f"  ECE:                 {metrics['ece']:.4f}")
    print(f"  Brier Score:         {metrics['brier_score']:.4f}")
    print(f"  Early Exit Rate:     {metrics['early_exit_rate']:.4f}")
    print(f"  Avg Debate Rounds:   {metrics['avg_debate_rounds']:.2f}")
    print(f"  Avg Tokens:          {metrics['avg_tokens']:.0f}")
    print(f"  Avg Latency (ms):    {metrics['avg_latency_ms']:.1f}")

    # Per-class F1
    f1s = metrics.get("per_class_f1", {})
    if f1s:
        print(f"\n  Per-Class F1:")
        for label, score in sorted(f1s.items()):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {label:30s}  {score:.4f}  [{bar}]")

    # Confusion matrix
    cm = metrics.get("confusion_matrix", {})
    if cm:
        print(f"\n  Confusion Matrix (true → predicted):")
        for true_label, preds in sorted(cm.items()):
            for pred_label, count in sorted(preds.items()):
                match = "✓" if true_label == pred_label else " "
                print(f"    {match} {true_label:25s} → {pred_label:25s} : {count}")

    # ── Part B: MetricsCalculator ──
    _sub_header("Part B: MetricsCalculator (NumPy-backed)")

    try:
        from dermarbiter.evaluation.metrics import MetricsCalculator

        calc = MetricsCalculator(records=results)
        extended = calc.compute_all()
        print(f"  Balanced Accuracy:   {extended.get('balanced_accuracy', 0):.4f}")
        print(f"  AUROC (macro):       {extended.get('auroc', 0):.4f}")
        print(f"  Avg Tool Calls:      {extended.get('avg_tool_calls', 0):.1f}")

        sens = extended.get("sensitivity", {})
        spec = extended.get("specificity", {})
        if sens:
            print(f"\n  Per-Class Sensitivity / Specificity:")
            for label in sorted(sens.keys()):
                s = sens.get(label, 0)
                sp = spec.get(label, 0)
                print(f"    {label:30s}  Sens={s:.4f}  Spec={sp:.4f}")
    except ImportError:
        print("  [SKIP] MetricsCalculator requires numpy — not available.")
    except Exception as e:
        print(f"  [SKIP] MetricsCalculator error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Fairness Analysis
# ═══════════════════════════════════════════════════════════════════════════

def section_4_fairness(results: List[Dict[str, Any]]) -> None:
    """
    Evaluate fairness across Fitzpatrick skin type subgroups using
    the FairnessAnalyzer with synthesised per-group data.
    """
    _banner("SECTION 4: Fairness Analysis")

    try:
        from dermarbiter.evaluation.fairness_analyzer import FairnessAnalyzer
    except ImportError:
        print("  [SKIP] FairnessAnalyzer requires numpy — not available.")
        return

    # Enrich results with fitzpatrick_type from patient_context if available
    # For this demo, we also synthesise extra records to demonstrate
    # multi-group analysis
    enriched = []
    for r in results:
        rec = dict(r)
        # Extract fitzpatrick from sample cases
        enriched.append(rec)

    # Add synthetic records across multiple Fitzpatrick types for richer demo
    synthetic_groups = [
        {"fitzpatrick_type": "I", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.88},
        {"fitzpatrick_type": "I", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.82},
        {"fitzpatrick_type": "II", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.85},
        {"fitzpatrick_type": "II", "predicted": "melanoma", "ground_truth": "BCC", "consensus_score": 0.70},
        {"fitzpatrick_type": "III", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.80},
        {"fitzpatrick_type": "III", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.75},
        {"fitzpatrick_type": "IV", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.72},
        {"fitzpatrick_type": "IV", "predicted": "BCC", "ground_truth": "melanoma", "consensus_score": 0.65},
        {"fitzpatrick_type": "V", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.68},
        {"fitzpatrick_type": "V", "predicted": "BCC", "ground_truth": "melanoma", "consensus_score": 0.60},
        {"fitzpatrick_type": "VI", "predicted": "melanoma", "ground_truth": "melanoma", "consensus_score": 0.62},
        {"fitzpatrick_type": "VI", "predicted": "BCC", "ground_truth": "melanoma", "consensus_score": 0.55},
    ]
    enriched.extend(synthetic_groups)

    print(f"  Records (including synthetic): {len(enriched)}")
    print(f"  Group key: fitzpatrick_type\n")

    analyzer = FairnessAnalyzer(records=enriched, group_key="fitzpatrick_type")
    report = analyzer.compute_all()

    # Per-group table
    print(f"  {'Group':<8s}  {'N':>4s}  {'Accuracy':>8s}  {'F1':>7s}  {'ECE':>7s}")
    print(f"  {'─' * 8}  {'─' * 4}  {'─' * 8}  {'─' * 7}  {'─' * 7}")

    for g in report.get("group_names", []):
        n = report["per_group_n"].get(g, 0)
        acc = report["per_group_accuracy"].get(g, 0.0)
        f1 = report["per_group_f1"].get(g, 0.0)
        ece = report.get("calibration_gap", {}).get("per_group_ece", {}).get(g, 0.0)
        print(f"  {g:<8s}  {n:>4d}  {acc:>8.4f}  {f1:>7.4f}  {ece:>7.4f}")

    # Disparity summary
    ag = report.get("accuracy_gap", {})
    eo = report.get("equalized_odds", {})
    dp = report.get("demographic_parity", {})

    print(f"\n  Accuracy Gap:")
    print(f"    Max Δ-Accuracy:  {ag.get('max_delta', 0):.4f}")
    print(f"    Best group:      {ag.get('max_group', 'N/A')}")
    print(f"    Worst group:     {ag.get('min_group', 'N/A')}")

    print(f"\n  Equalized Odds:")
    print(f"    Max TPR disp.:   {eo.get('max_tpr_disparity', 0):.4f}")
    print(f"    Max FPR disp.:   {eo.get('max_fpr_disparity', 0):.4f}")
    satisfied = "✓ YES" if eo.get("satisfied", False) else "✗ NO"
    print(f"    Satisfied:       {satisfied}")

    print(f"\n  Demographic Parity:")
    print(f"    Max disparity:   {dp.get('max_disparity', 0):.4f}")
    satisfied = "✓ YES" if dp.get("satisfied", False) else "✗ NO"
    print(f"    Satisfied:       {satisfied}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Ablation Preview
# ═══════════════════════════════════════════════════════════════════════════

def section_5_ablation() -> None:
    """
    Show how AblationRunner can be configured for systematic ablation
    studies, and run one quick agent-ablation variant.
    """
    _banner("SECTION 5: Ablation Preview")

    from dermarbiter.experiments.ablation import (
        AblationRunner,
        AblationConfig,
        AGENT_ABLATION,
        TOOL_ABLATION,
        ROUND_ABLATION,
    )

    # ── Part A: Show all variant configurations ──
    _sub_header("Part A: Ablation Variant Configurations")

    print("  Agent Ablation variants:")
    for role in ["specialist", "generalist", "skeptic"]:
        cfg = AblationConfig(
            ablation_type=AGENT_ABLATION,
            variant_name=f"no_{role}",
            removed_agents=[role],
        )
        print(f"    • {cfg.variant_name:25s}  remove=[{role}]")

    print("\n  Tool Ablation variants (examples):")
    tool_examples = ["panderm_classifier", "make_annotator", "case_rag",
                     "guideline_rag", "fairness_probe"]
    for tool in tool_examples:
        cfg = AblationConfig(
            ablation_type=TOOL_ABLATION,
            variant_name=f"no_{tool}",
            removed_tools=[tool],
        )
        print(f"    • {cfg.variant_name:25s}  remove=[{tool}]")

    print("\n  Round Ablation variants:")
    for rounds in [1, 2, 3, 5]:
        cfg = AblationConfig(
            ablation_type=ROUND_ABLATION,
            variant_name=f"max_rounds_{rounds}",
            max_rounds=rounds,
        )
        print(f"    • {cfg.variant_name:25s}  max_rounds={rounds}")

    # ── Part B: Run one quick agent ablation ──
    _sub_header("Part B: Quick Agent Ablation (no_skeptic)")

    data_path = str(_PROJECT_ROOT / "data" / "sample_cases.jsonl")
    output_path = str(_PROJECT_ROOT / "results" / "demo_ablation.jsonl")

    t0 = time.time()
    runner = AblationRunner(
        config_path="configs/",
        data_path=data_path,
        ablation_type=AGENT_ABLATION,
        output_path=output_path,
        mock=True,
        max_cases=2,  # Quick: only 2 cases
    )
    ablation_results = runner.run()
    t1 = time.time()

    # Summarise by variant
    from collections import defaultdict
    variant_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "tokens": 0}
    )
    for r in ablation_results:
        v = r.get("variant", "unknown")
        variant_stats[v]["total"] += 1
        variant_stats[v]["tokens"] += r.get("total_tokens", 0)
        if r.get("predicted", "").strip().lower() == r.get("ground_truth", "").strip().lower():
            variant_stats[v]["correct"] += 1

    print(f"  {'Variant':<25s}  {'Correct':<8s}  {'Total':<6s}  {'Acc':>6s}  {'Tokens':>7s}")
    print(f"  {'─' * 25}  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 7}")
    for variant, stats in sorted(variant_stats.items()):
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {variant:<25s}  {stats['correct']:<8d}  {stats['total']:<6d}  "
              f"{acc:>6.2f}  {stats['tokens']:>7d}")

    print(f"\n  Total variants: {len(variant_stats)}")
    print(f"  Total results:  {len(ablation_results)}")
    print(f"  Time: {(t1 - t0):.2f}s")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Architecture Visualization
# ═══════════════════════════════════════════════════════════════════════════

def section_6_architecture() -> None:
    """
    Print the full DermArbiter pipeline architecture as ASCII art.
    """
    _banner("SECTION 6: Pipeline Architecture")

    architecture = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    DermArbiter Pipeline                      │
    │              Multi-Agent Diagnostic Debate Panel             │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
    │  Clinical    │     │  Dermoscopic │     │   Patient     │
    │  Query       │────▶│  Image       │────▶│   Context     │
    └──────┬──────┘     └──────┬──────┘     └──────┬───────┘
           │                   │                    │
           └───────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   BlackboardState    │
                    │  (Shared Workspace)  │
                    └──────────┬──────────┘
                               │
    ╔══════════════════════════╪══════════════════════════════╗
    ║  PHASE 1: Plan & Probe  │                              ║
    ║                          ▼                              ║
    ║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ ║
    ║  │Specialist│  │Generalist│  │ Skeptic  │  │Moderator║ ║
    ║  │ propose  │  │ propose  │  │ (no tool)│  │ propose │ ║
    ║  └────┬─────┘  └────┬─────┘  └──────────┘  └───┬────┘ ║
    ║       │              │                          │      ║
    ║       └──────────────┼──────────────────────────┘      ║
    ║                      ▼                                 ║
    ║          ┌──────────────────────┐                      ║
    ║          │  ToolRegistry        │  9 diagnostic tools  ║
    ║          │  run_batch()         │  (PanDerm, MAKE, …)  ║
    ║          └──────────┬───────────┘                      ║
    ║                     ▼                                  ║
    ║          ┌──────────────────────┐                      ║
    ║          │  Evidence Cards      │                      ║
    ║          │  → Blackboard        │                      ║
    ║          └──────────────────────┘                      ║
    ╚════════════════════════════════════════════════════════╝
                               │
    ╔══════════════════════════╪══════════════════════════════╗
    ║  PHASE 2: Independent    │                             ║
    ║           Reading        ▼                             ║
    ║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐║
    ║  │Specialist│  │Generalist│  │ Skeptic  │  │Moderator║║
    ║  │  Brief   │  │  Brief   │  │  Brief   │  │  Brief  ║║
    ║  │ conf=.85 │  │ conf=.68 │  │ conf=.55 │  │ conf=.79║║
    ║  └──────────┘  └──────────┘  └──────────┘  └────────┘║
    ╚════════════════════════════════════════════════════════╝
                               │
    ╔══════════════════════════╪══════════════════════════════╗
    ║  PHASE 3: Reveal &       │                             ║
    ║           Critique       ▼                             ║
    ║          ┌──────────────────────┐                      ║
    ║          │  Moderator evaluates │                      ║
    ║          │  consensus / gating  │                      ║
    ║          └──────────┬───────────┘                      ║
    ║                     │                                  ║
    ║            ┌────────┴────────┐                         ║
    ║            ▼                 ▼                         ║
    ║     ┌──────────┐     ┌────────────┐                   ║
    ║     │ Consensus│     │ No Consensus│                   ║
    ║     │ → Phase 5│     │ → Phase 4  │                   ║
    ║     └──────────┘     └────────────┘                   ║
    ╚════════════════════════════════════════════════════════╝
                               │
    ╔══════════════════════════╪══════════════════════════════╗
    ║  PHASE 4: Targeted       │                             ║
    ║           Debate         ▼                             ║
    ║      max_rounds=3, token budget=50K                    ║
    ║                                                        ║
    ║   Round 1: Specialist → Generalist → Skeptic           ║
    ║   Round 2: Specialist → Generalist → Skeptic           ║
    ║   Round 3: Specialist → Generalist → Skeptic           ║
    ║                                                        ║
    ║   Per-turn: argument + rebuttal, token-limited         ║
    ╚════════════════════════════════════════════════════════╝
                               │
    ╔══════════════════════════╪══════════════════════════════╗
    ║  PHASE 5: Synthesis      │                             ║
    ║                          ▼                             ║
    ║  ┌─────────────────────────────────────────────────┐   ║
    ║  │ 1. Weighted rank aggregation                    │   ║
    ║  │    specialist_weight=1.2 × rank_weights=[1,.6,.3]│  ║
    ║  │ 2. Consensus score (primary dx agreement)       │   ║
    ║  │ 3. Dissent notes from disagreement flags        │   ║
    ║  │ 4. Moderator → Final Clinical Report            │   ║
    ║  └─────────────────────────────────────────────────┘   ║
    ╚════════════════════════════════════════════════════════╝
                               │
                    ┌──────────▼──────────┐
                    │   Final Outputs      │
                    │  • Ranked diagnoses   │
                    │  • Consensus score    │
                    │  • Clinical report    │
                    │  • Dissent notes      │
                    │  • Telemetry          │
                    └──────────────────────┘

    Agents:
      ┌─────────────┬────────────┬──────────┬────────────────┐
      │ Specialist  │ Generalist │ Skeptic  │ Moderator      │
      │ Gemini 2.5  │ MedGemma   │ Qwen3-8B │ Gemini 2.5     │
      │ Flash       │ 4B         │ (no tool)│ Flash          │
      │ ★ High conf │ ★ Broad DDx│ ★ Critic │ ★ Synthesizer  │
      └─────────────┴────────────┴──────────┴────────────────┘

    Tools (9 total):
      ┌───────────────────┬─────────────────┬──────────────────┐
      │ panderm_classifier│ make_annotator  │ dermogpt_vqa     │
      │ general_vqa       │ guideline_rag   │ case_rag         │
      │ ontology_graph    │ fairness_probe ★│ uncertainty_probe★│
      └───────────────────┴─────────────────┴──────────────────┘
      ★ = Novel contribution
"""
    for line in architecture.strip().split("\n"):
        print(line)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  DERMARBITER — FULL PIPELINE DEMO".center(68) + "║")
    print("║" + "  Mock Mode · CPU Only · No API Keys Required".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    t_global = time.time()

    # Section 1: Single case through the full pipeline
    state = section_1_single_case()

    # Section 2: Batch benchmark on all 5 sample cases
    results = section_2_batch_benchmark()

    # Section 3: Comprehensive metrics analysis
    section_3_metrics(results)

    # Section 4: Fairness analysis across skin types
    section_4_fairness(results)

    # Section 5: Ablation study preview
    section_5_ablation()

    # Section 6: Architecture visualisation
    section_6_architecture()

    # ── Final Summary ──
    elapsed = time.time() - t_global
    _banner("DEMO COMPLETE", char="═")
    print(f"  All 6 sections executed successfully ✓")
    print(f"  Total wall-clock time: {elapsed:.2f}s")
    print(f"  Pipeline mode: Mock (CPU-only)")
    print(f"  For GPU execution: python scripts/run_e2e_gpu.py --mock")
    print()


if __name__ == "__main__":
    main()
